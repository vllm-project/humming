import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

import pytest
import torch

import humming.jit.compiler as compiler_module
import humming.utils.jit as jit_utils
from humming.jit.compiler import Compiler


class _FakeCompiler(Compiler):
    compile_calls = 0

    @classmethod
    def signature(cls):
        return "fake-compiler"

    @classmethod
    def get_flags(cls, sm_version, disable_fast_math=False):
        return [f"sm={sm_version}"]

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr, flags):
        cls.compile_calls += 1
        (Path(cache_dirname) / "kernel_tmp.cubin").write_bytes(b"fake-cubin")
        return 0, "stdout", "stderr"


def test_compiler_publishes_cache_while_holding_lock(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    lock_active = False

    class TrackingLock:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            nonlocal lock_active
            assert not lock_active
            lock_active = True

        def __exit__(self, exc_type, exc, traceback):
            nonlocal lock_active
            lock_active = False

    real_replace = os.replace

    def checked_replace(source, destination):
        assert lock_active, "the final cubin must be published before releasing the cache lock"
        return real_replace(source, destination)

    monkeypatch.setattr(jit_utils, "get_humming_cache_dir", lambda: cache_dir.as_posix())
    monkeypatch.setattr(
        jit_utils,
        "get_humming_lock_filename",
        lambda name: (lock_dir / f"{name}.lock").as_posix(),
    )
    monkeypatch.setattr(Compiler, "cuh_last_update_time", staticmethod(lambda: "headers"))
    monkeypatch.setattr(compiler_module, "FileLock", TrackingLock)
    monkeypatch.setattr(compiler_module.os, "replace", checked_replace)
    _FakeCompiler.compile_calls = 0

    first = _FakeCompiler.compile("source", "90a", "kernel")
    second = _FakeCompiler.compile("source", "90a", "kernel")

    assert first == second
    assert Path(first).read_bytes() == b"fake-cubin"
    assert _FakeCompiler.compile_calls == 1
    assert not list(cache_dir.rglob("kernel_tmp.cubin"))


def test_precompiled_manifest_uses_content_hashes(tmp_path, monkeypatch):
    arch = jit_utils.get_native_arch()
    assert arch is not None

    package_dir = tmp_path / "humming"
    native_dir = package_dir / "_native" / arch
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    native_dir.mkdir(parents=True)
    source = source_dir / "helper.cpp"
    artifact = native_dir / "helper"
    source.write_text("version one")
    artifact.write_bytes(b"native version one")

    jit_utils.write_precompiled_artifact_manifest(native_dir, {artifact.name: source})
    monkeypatch.setattr(jit_utils, "__file__", (package_dir / "utils/jit.py").as_posix())

    assert jit_utils.get_precompiled_artifact_path(source, artifact.name) == artifact

    source_stat = source.stat()
    os.utime(
        source,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
    )
    assert jit_utils.get_precompiled_artifact_path(source, artifact.name) == artifact

    source_stat = source.stat()
    source.write_text("version two")
    os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    assert jit_utils.get_precompiled_artifact_path(source, artifact.name) is None


@pytest.mark.parametrize("is_directory", (False, True))
def test_precompiled_manifest_survives_package_relocation(tmp_path, monkeypatch, is_directory):
    package_dir = tmp_path / "build" / "humming"
    source_dir = package_dir / "csrc" / "launcher"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "launcher.cpp"
    source_file.write_bytes(b"native source\n")
    source = source_dir if is_directory else source_file
    native_dir = package_dir / "_native" / jit_utils.get_native_arch()
    native_dir.mkdir(parents=True)
    artifact = native_dir / "helper"
    artifact.write_bytes(b"native artifact")
    jit_utils.write_precompiled_artifact_manifest(native_dir, {artifact.name: source})

    installed_dir = tmp_path / "installed" / "humming"
    shutil.copytree(package_dir, installed_dir)
    monkeypatch.setattr(jit_utils, "__file__", str(installed_dir / "utils" / "jit.py"))
    installed_source = installed_dir / source.relative_to(package_dir)
    installed_artifact = installed_dir / artifact.relative_to(package_dir)
    assert jit_utils.get_precompiled_artifact_path(installed_source, artifact.name) == installed_artifact


@pytest.mark.parametrize(
    "is_windows, expected",
    [
        (False, ("libhumming_launcher.so", "libcubinpatch.so", "_device_info.abi3.so", "nvrtc_compile")),
        (True, ("humming_launcher.dll", "cubinpatch.dll", "_device_info.pyd", "nvrtc_compile.exe")),
    ],
)
def test_native_artifact_names(monkeypatch, is_windows, expected):
    monkeypatch.setattr(jit_utils, "IS_WINDOWS", is_windows)
    components = ("launcher", "cubin_patcher", "device_info", "nvrtc_compile")
    assert tuple(jit_utils.get_native_artifact_name(name) for name in components) == expected


@pytest.mark.parametrize("is_windows", (False, True))
@pytest.mark.parametrize("cuda_variable", ("CUDA_HOME", "CUDA_PATH"))
@pytest.mark.parametrize("name", ("nvcc", "nvcc.exe", "nvcc.EXE"))
def test_cuda_command_executable_suffix(tmp_path, monkeypatch, is_windows, cuda_variable, name):
    monkeypatch.setattr(jit_utils, "IS_WINDOWS", is_windows)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv(f"CUDA_{name.upper()}_PATH", raising=False)
    monkeypatch.setenv(cuda_variable, str(tmp_path / "CUDA Toolkit"))
    expected_name = "nvcc.exe" if is_windows and name == "nvcc" else name
    expected = tmp_path / "CUDA Toolkit" / "bin" / expected_name
    assert jit_utils.get_cuda_command_path.__wrapped__(name) == str(expected)


@pytest.mark.parametrize("name", (r"C:\CUDA Toolkit\bin\nvcc.exe", "C:/CUDA Toolkit/bin/nvcc.exe"))
def test_cuda_command_preserves_explicit_path(monkeypatch, name):
    monkeypatch.setattr(jit_utils, "IS_WINDOWS", True)
    assert jit_utils.get_cuda_command_path.__wrapped__(name) == name


def test_windows_cuda_discovery(tmp_path, monkeypatch):
    from humming.utils import cuda

    toolkit = tmp_path / "CUDA Toolkit"
    (toolkit / "include").mkdir(parents=True)
    (toolkit / "include" / "cuda.h").touch()
    (toolkit / "bin").mkdir()
    nvcc = toolkit / "bin" / "nvcc.exe"
    nvcc.touch()
    nvcc.chmod(nvcc.stat().st_mode | 0o111)
    (toolkit / "version.json").write_text('{"cuda": {"version": "13.4.0"}}')
    monkeypatch.setattr(jit_utils, "IS_WINDOWS", True)
    monkeypatch.setenv("CUDA_PATH", str(toolkit))
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(cuda, "_find_nvidia_pypi_cuda_paths", lambda: [])

    environments = cuda.find_all_cuda_paths.__wrapped__()
    (environment,) = [value for value in environments if value["path"] == str(toolkit)]
    assert environment["major"] == 13
    assert environment["minor"] == 4
    assert environment["binaries"]["nvcc"] == str(toolkit / "bin" / "nvcc.exe")


def test_cuda_discovery_rejects_cpu_only_torch(monkeypatch):
    from humming.utils.cuda import filter_cuda_paths

    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch is required"):
        filter_cuda_paths()


@pytest.mark.parametrize("component_dir", ("", "cuda_nvrtc"))
@pytest.mark.parametrize(
    "native_arch, directory_arch",
    [
        ("aarch64", "arm64"),
        ("x86_64", "x64"),
        ("x86_64", "x86_64"),
        ("x86_64", ""),
    ],
)
def test_windows_cuda_libraries(tmp_path, monkeypatch, component_dir, native_arch, directory_arch):
    from humming.utils import cuda, nvrtc

    root = tmp_path / component_dir
    library_dir = root / "lib" / directory_arch
    binary_dir = root / "bin" / directory_arch
    library_dir.mkdir(parents=True)
    binary_dir.mkdir(parents=True)
    (library_dir / "cuda.lib").touch()
    (binary_dir / "nvrtc64_130_0.dll").touch()
    (binary_dir / "nvrtc-builtins64_134.dll").touch()
    environment = {"path": str(tmp_path), "include_paths": []}
    monkeypatch.setattr(jit_utils, "IS_WINDOWS", True)
    monkeypatch.setattr(jit_utils, "get_native_arch", lambda: native_arch)
    monkeypatch.setattr(cuda, "filter_cuda_paths", lambda **kwargs: environment)
    monkeypatch.setattr(nvrtc, "filter_cuda_paths", lambda **kwargs: environment)
    assert cuda.get_cuda_library_path("cuda.lib") == library_dir / "cuda.lib"
    assert nvrtc.get_nvrtc_library_path() == str(binary_dir / "nvrtc64_130_0.dll")
    (library_dir / "cuda.lib").unlink()
    other_arch = "x86_64" if native_arch == "aarch64" else "arm64"
    other_library_dir = root / "lib" / other_arch
    other_library_dir.mkdir(parents=True, exist_ok=True)
    (other_library_dir / "cuda.lib").touch()
    with pytest.raises(RuntimeError, match="Could not locate cuda.lib"):
        cuda.get_cuda_library_path("cuda.lib")
    (binary_dir / "nvrtc64_130_0.dll").unlink()
    other_binary_dir = root / "bin" / other_arch
    other_binary_dir.mkdir(parents=True, exist_ok=True)
    (other_binary_dir / "nvrtc64_130_0.dll").touch()
    with pytest.raises(RuntimeError, match="Could not locate the NVRTC library"):
        nvrtc.get_nvrtc_library_path()


def test_msvc_build_keeps_intermediates_out_of_package(tmp_path, monkeypatch):
    commands = []
    source = tmp_path / "source.cpp"
    source.touch()
    output = tmp_path / "native" / "helper.dll"

    def run(command, **kwargs):
        commands.append(command)
        binary = Path(next(value.removeprefix("/OUT:") for value in command if value.startswith("/OUT:")))
        assert binary.parent == Path(kwargs["cwd"])
        binary.write_bytes(b"native binary")
        (binary.parent / "source.obj").touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(jit_utils.subprocess, "run", run)
    jit_utils.build_msvc_native(source, output, "cl", include_paths=["include path"])
    assert output.read_bytes() == b"native binary"
    assert list(output.parent.iterdir()) == [output]
    assert "/LD" in commands[0]
    assert "/Iinclude path" in commands[0]
    assert not any(value.startswith("-") for value in commands[0])


def test_msvc_build_failure_preserves_existing_artifact(tmp_path, monkeypatch):
    source = tmp_path / "source.cpp"
    source.touch()
    output = tmp_path / "helper.dll"
    output.write_bytes(b"previous build")
    monkeypatch.setattr(
        jit_utils.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "compiler error"),
    )
    with pytest.raises(RuntimeError, match="compiler error"):
        jit_utils.build_msvc_native(source, output, "cl", include_paths=[])
    assert output.read_bytes() == b"previous build"


def test_native_mapped_file(tmp_path):
    compiler = jit_utils.get_cpp_compiler()
    include_dir = Path(jit_utils.__file__).parents[1] / "csrc" / "launcher"
    source = tmp_path / "mapped_file_test.cpp"
    source.write_text(
        '#include "mapped_file.h"\n'
        "int main(int argc, char **argv) {\n"
        "  MappedFile file(argv[1]);\n"
        '  if (file.size() != 6 || std::memcmp(file.data(), "cubin!", 6) != 0) return 1;\n'
        "  for (int i = 2; i < argc; ++i) {\n"
        "    try { MappedFile invalid(argv[i]); return 2; }\n"
        "    catch (const std::runtime_error &) {}\n"
        "  }\n"
        "  return 0;\n"
        "}\n"
    )
    binary = tmp_path / ("mapped_file_test.exe" if jit_utils.IS_WINDOWS else "mapped_file_test")
    if jit_utils.IS_WINDOWS:
        jit_utils.build_msvc_native(source, binary, compiler, include_paths=[str(include_dir)], shared=False)
    else:
        subprocess.run(
            [compiler, "-std=c++17", f"-I{include_dir}", str(source), "-o", str(binary)],
            check=True,
        )
    data = tmp_path / "file with spaces"
    data.write_bytes(b"cubin!")
    empty = tmp_path / "empty"
    empty.touch()
    subprocess.run([str(binary), str(data), str(empty), str(tmp_path / "missing")], check=True)


@pytest.fixture(scope="module")
def mock_device_info(tmp_path_factory):
    from humming.utils.device import _get_cuda_include_paths

    source = Path(__file__).parent / "native" / "device_info_mock.cpp"
    csrc_dir = Path(jit_utils.__file__).parents[1] / "csrc"
    output = tmp_path_factory.mktemp("device-info") / jit_utils.get_native_artifact_name("device_info")
    compiler = jit_utils.get_cpp_compiler()
    include_paths = [str(csrc_dir), sysconfig.get_paths()["include"], *_get_cuda_include_paths()]
    if jit_utils.IS_WINDOWS:
        jit_utils.build_msvc_native(
            source,
            output,
            compiler,
            include_paths=include_paths,
            link_args=[str(Path(sys.base_prefix) / "libs" / "python3.lib")],
        )
    else:
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-shared",
                "-fPIC",
                *[f"-I{path}" for path in include_paths],
                str(source),
                "-o",
                str(output),
            ],
            check=True,
        )
    spec = importlib.util.spec_from_file_location("humming_test._device_info", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _ = module._DeviceInfo(0).memory_bus_width
    is_windows_arm64 = jit_utils.IS_WINDOWS and jit_utils.get_native_arch() == "aarch64"
    expected_warnings = 2 if is_windows_arm64 else 0
    assert len(recorded) == expected_warnings
    for warning in recorded:
        assert warning.category is RuntimeWarning
        assert "using GB10-reference clocks and bus width" in str(warning.message)
    return module


@pytest.mark.parametrize(
    "device_index, sm_version, bus_width, sm_count",
    [(0, 121, 0, 48), (1, 121, 128, 40), (2, 120, 0, 32), (3, 89, 0, 24), (4, 121, 0, 16), (5, 121, -1, 24)],
)
def test_windows_device_metadata_fallback(mock_device_info, device_index, sm_version, bus_width, sm_count):
    info = mock_device_info._DeviceInfo(device_index)
    is_windows_arm64 = jit_utils.IS_WINDOWS and jit_utils.get_native_arch() == "aarch64"
    use_fallback = is_windows_arm64 and sm_version == 121 and bus_width == 0
    expected_bus_width = 256 if use_fallback else bus_width
    expected_memory_clock = 8533000 if use_fallback else 1048576
    expected_sm_clock = 3003000 if use_fallback else 1048576
    assert info.sm_version == sm_version
    assert info.sm_count == sm_count
    assert info.memory_bus_width == expected_bus_width
    assert info.memory_clock_khz == expected_memory_clock
    assert info.sm_clock_khz == expected_sm_clock
    multiplier = 1 if sm_version == 121 else 2
    expected_bandwidth = expected_memory_clock * multiplier * expected_bus_width / 8 / 1e6
    assert info.memory_bandwidth_gbps == pytest.approx(expected_bandwidth)
    ops_per_clock = 512 if sm_version == 89 else 1024
    assert info.tensorcore_tops["float16"] == pytest.approx(
        sm_count * ops_per_clock * expected_sm_clock / 1e9
    )


def test_precompiled_native_helpers(tmp_path):
    from humming.ops.utils import _get_precompiled_launcher_path, init_humming_launcher
    from humming.utils import cubin, device, nvrtc

    package_dir = Path(jit_utils.__file__).parents[1]
    if not (package_dir / "_native").is_dir():
        pytest.skip("requires precompiled native artifacts")
    sources = {
        "launcher": "launcher",
        "device_info": "device_info.cpp",
        "cubin_patcher": "patch_cubin.cpp",
        "nvrtc_compile": "nvrtc_compile.cpp",
    }
    for component, source in sources.items():
        artifact = jit_utils.get_precompiled_artifact_path(
            package_dir / "csrc" / source, jit_utils.get_native_artifact_name(component)
        )
        assert artifact is not None, f"Missing or stale precompiled {component}"
    assert _get_precompiled_launcher_path() is not None
    init_humming_launcher()
    assert device._get_device_info_extension()._DeviceInfo(-1) is not None
    with pytest.raises(RuntimeError, match="rc=3"):
        cubin.patch_cubin_bytes(b"not a cubin", "cvt_e3m4")

    source = tmp_path / "smoke.cu"
    source.write_text('extern "C" __global__ void smoke(float *x) { x[0] = 1.0f; }\n')
    output = tmp_path / "smoke.cubin"
    environment = os.environ.copy()
    if jit_utils.IS_WINDOWS:
        environment = {
            name: value for name, value in environment.items() if not name.upper().startswith("CUDA")
        }
        environment["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    result = subprocess.run(
        [
            nvrtc.may_build_nvrtc_compile_binary(),
            "--nvrtc-path",
            nvrtc.get_nvrtc_library_path(),
            "--input",
            str(source),
            "--output",
            str(output),
            "--",
            "--gpu-architecture=sm_80",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"NVRTC exited with {result.returncode}:\n{result.stdout}\n{result.stderr}"
    assert cubin.get_cubin_kernel_names(output) == ["smoke"]


def test_background_launcher_build_skips_usable_precompiled_artifact(monkeypatch):
    from humming.ops import utils

    monkeypatch.setenv("HUMMING_DISABLE_PARALLEL_BUILD", "0")
    monkeypatch.setattr(utils, "_resolve_use_torch_stable_api", lambda: True)
    monkeypatch.setattr(utils, "_get_precompiled_launcher_path", lambda: Path("launcher.so"))

    def fail_if_spawned(*args, **kwargs):
        raise AssertionError("a usable precompiled launcher should skip the helper")

    monkeypatch.setattr(jit_utils, "popen_and_reap", fail_if_spawned)

    assert utils.build_humming_launcher_in_bg() is None


def test_background_launcher_build_spawns_without_usable_precompiled_artifact(monkeypatch):
    from humming.ops import utils

    monkeypatch.setenv("HUMMING_DISABLE_PARALLEL_BUILD", "0")
    monkeypatch.setattr(utils, "_resolve_use_torch_stable_api", lambda: True)
    monkeypatch.setattr(utils, "_get_precompiled_launcher_path", lambda: None)
    spawned = []
    monkeypatch.setattr(jit_utils, "popen_and_reap", lambda *args, **kwargs: spawned.append(args))

    utils.build_humming_launcher_in_bg()

    assert len(spawned) == 1


def test_background_launcher_build_spawns_when_stable_api_is_disabled(monkeypatch):
    from humming.ops import utils

    monkeypatch.setenv("HUMMING_DISABLE_PARALLEL_BUILD", "0")
    monkeypatch.setattr(utils, "_resolve_use_torch_stable_api", lambda: False)
    monkeypatch.setattr(utils, "_get_precompiled_launcher_path", lambda: Path("launcher.so"))
    spawned = []
    monkeypatch.setattr(jit_utils, "popen_and_reap", lambda *args, **kwargs: spawned.append(args))

    utils.build_humming_launcher_in_bg()

    assert len(spawned) == 1


def test_background_nvrtc_build_skips_usable_precompiled_artifact(monkeypatch):
    from humming.utils import nvrtc

    monkeypatch.setenv("HUMMING_DISABLE_PARALLEL_BUILD", "0")
    monkeypatch.setattr(
        nvrtc, "_get_precompiled_nvrtc_compile_binary_path", lambda: Path("nvrtc_compile")
    )

    def fail_if_spawned(*args, **kwargs):
        raise AssertionError("a usable precompiled NVRTC helper should skip the child process")

    monkeypatch.setattr(jit_utils, "popen_and_reap", fail_if_spawned)

    assert nvrtc.build_nvrtc_compile_binary_in_bg() is None


def test_background_nvrtc_build_spawns_without_usable_precompiled_artifact(monkeypatch):
    from humming.utils import nvrtc

    monkeypatch.setenv("HUMMING_DISABLE_PARALLEL_BUILD", "0")
    monkeypatch.setattr(nvrtc, "_get_precompiled_nvrtc_compile_binary_path", lambda: None)
    spawned = []
    monkeypatch.setattr(jit_utils, "popen_and_reap", lambda *args, **kwargs: spawned.append(args))

    nvrtc.build_nvrtc_compile_binary_in_bg()

    assert len(spawned) == 1


@pytest.mark.skipif(not jit_utils.IS_WINDOWS, reason="requires Windows DLL loading")
@pytest.mark.parametrize("unicode_library_path", (False, True))
@pytest.mark.parametrize("emit_ptx", (False, True))
def test_nvrtc_compile_unicode_paths(tmp_path, unicode_library_path, emit_ptx):
    from humming.utils import cubin, nvrtc

    work_dir = tmp_path / "NVRTC \u00e9\u4e2d\u6587"
    work_dir.mkdir()
    library = Path(nvrtc.get_nvrtc_library_path())
    if unicode_library_path:
        library_dir = work_dir / "bin"
        library_dir.mkdir()
        for dependency in library.parent.glob("nvrtc*.dll"):
            shutil.copyfile(dependency, library_dir / dependency.name)
        library = library_dir / library.name

    source = work_dir / "\u6e90.cu"
    header = work_dir / "\u5934.h"
    output = work_dir / ("\u8f93\u51fa.ptx" if emit_ptx else "\u8f93\u51fa.cubin")
    log = work_dir / "\u65e5\u5fd7.log"
    source.write_text(
        '#include "value.h"\nextern "C" __global__ void smoke(float *x) { x[0] = VALUE; }\n',
        encoding="utf-8",
    )
    header.write_text("#define VALUE 1.0f\n", encoding="utf-8")
    environment = {
        name: value for name, value in os.environ.items() if not name.upper().startswith("CUDA")
    }
    environment["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    result = subprocess.run(
        [
            nvrtc.may_build_nvrtc_compile_binary(),
            "--nvrtc-path",
            str(library),
            "--input",
            str(source),
            "--output",
            str(output),
            "--log",
            str(log),
            "--header",
            f"value.h={header}",
            *(["--ptx"] if emit_ptx else []),
            "--",
            "--gpu-architecture=sm_80",
        ],
        env=environment,
        capture_output=True,
    )
    assert result.returncode == 0, f"NVRTC exited with {result.returncode}:\n{result.stdout}\n{result.stderr}"
    assert log.is_file()
    if emit_ptx:
        assert ".entry smoke(" in output.read_text(encoding="utf-8")
    else:
        assert cubin.get_cubin_kernel_names(output) == ["smoke"]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("tuning_source", ("heuristic", "batch_invariant", "heuristic+batch_invariant"))
def test_launcher_runs_device_local_kernels(monkeypatch, tuning_source):
    from humming import dtypes
    from humming.config import ComputeConfig, GemmType, LayerConfig
    from humming.kernel.humming import HummingKernel
    from humming.testing import KernelTestCase, KernelTestRunner, assert_kernel_test_shape_coverage

    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", tuning_source)
    compile_batches = []
    compile_many = HummingKernel.compile_many

    def record_compile_batch(kernel_specs, device):
        compile_batches.append([config["use_batch_invariant"] for _, config in kernel_specs])
        return compile_many(kernel_specs, device)

    monkeypatch.setattr(HummingKernel, "compile_many", record_compile_batch)

    def make_case():
        return KernelTestCase(
            name="multi-device-smoke",
            layer_config=LayerConfig(
                shape_n=64,
                shape_k=32,
                a_dtype=dtypes.bfloat16,
                b_dtype=dtypes.uint4,
                c_dtype=dtypes.bfloat16,
                bs_dtype=dtypes.bfloat16,
            ),
            compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
            seed=2026,
        )

    for device_index in range(2):
        with torch.cuda.device(device_index):
            runner = KernelTestRunner(make_case())
            results = runner.run((1,))
            torch.cuda.synchronize(device_index)
            assert_kernel_test_shape_coverage(results, (1,))
            assert len(compile_batches) == device_index + 1
            expected_modes = [source == "batch_invariant" for source in tuning_source.split("+")]
            assert compile_batches[-1] == expected_modes
            assert not runner.compute_config.use_batch_invariant
            for result in results:
                torch.testing.assert_close(
                    result.outputs,
                    result.outputs_ref,
                    rtol=runner.test_case.rtol,
                    atol=runner.test_case.atol,
                )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_kernel_runtime_instances_are_context_local():
    from humming import dtypes
    from humming.kernel.process_input import ProcessInputKernel
    from humming.ops import process_input

    def make_kernel():
        return ProcessInputKernel(
            input_dtype=dtypes.float32,
            hidden_size=32,
            quant_group_size=32,
            hadamard_block_size=32,
            threads_per_task=32,
            values_per_thread=1,
            quant_mode="none",
            use_tile_partition=True,
        )

    values = torch.randn((2, 32), dtype=torch.float32)
    input0 = values.to("cuda:0")
    input1 = values.to("cuda:1")

    with torch.cuda.device(0):
        output0 = process_input(input0, hadamard_block_size=32)[0]
        kernel0 = make_kernel()
    with torch.cuda.device(1):
        output1 = process_input(input1, hadamard_block_size=32)[0]
        kernel1 = make_kernel()

    assert kernel0 is not kernel1
    torch.testing.assert_close(output0.cpu(), output1.cpu(), rtol=0, atol=0)


def test_nvrtc_signature_uses_subprocess_library(monkeypatch):
    import ctypes

    loaded_paths = []

    class VersionFunction:
        def __call__(self, major, minor):
            ctypes.cast(major, ctypes.POINTER(ctypes.c_int))[0] = 13
            ctypes.cast(minor, ctypes.POINTER(ctypes.c_int))[0] = 2
            return 0

    class Library:
        nvrtcVersion = VersionFunction()

    def load_library(path):
        loaded_paths.append(path)
        return Library()

    monkeypatch.setattr(compiler_module, "get_nvrtc_library_path", lambda: "/toolkit/libnvrtc.so")
    monkeypatch.setattr(compiler_module.ctypes, "CDLL", load_library)
    compiler_module.NVRTCCompiler._get_version.cache_clear()
    try:
        assert compiler_module.NVRTCCompiler.signature() == "nvrtc+13.2"
        assert compiler_module.NVRTCCompiler.signature() == "nvrtc+13.2"
        assert loaded_paths == ["/toolkit/libnvrtc.so"]
    finally:
        compiler_module.NVRTCCompiler._get_version.cache_clear()


@pytest.mark.parametrize("limit", (1, 3, 32))
def test_compile_many_obeys_worker_limit(monkeypatch, limit):
    from contextlib import nullcontext

    from humming.jit import runtime

    worker_counts = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            worker_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, specs):
            return map(function, specs)

    monkeypatch.setattr(runtime, "get_parallel_build_workers", lambda: limit)
    monkeypatch.setattr(runtime, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(runtime.torch.cuda, "device", lambda device: nullcontext())
    specs = [(lambda value: value, {"value": value}) for value in range(5)]
    assert runtime.KernelRuntime.compile_many(specs, device=0) == list(range(5))
    assert worker_counts == ([] if limit == 1 else [min(limit, len(specs))])
    assert runtime.KernelRuntime.compile_many([], device=0) == []

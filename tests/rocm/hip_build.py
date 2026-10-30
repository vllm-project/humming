"""Standalone hipcc build helper for the ROCm MFMA kernels.

Compiles a HIP source (with host-side ``extern "C"`` launchers) into a shared
library and loads it with ctypes. Builds are cached by a hash of the source,
the included Humming headers and the compiler flags. ``torch`` must be imported
first so that the HIP runtime it ships is the one the library binds to.
"""

import ctypes
import hashlib
import os
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
INCLUDE_DIR = REPO_ROOT / "humming" / "include"
BUILD_DIR = pathlib.Path(os.environ.get("HUMMING_ROCM_BUILD_DIR", REPO_ROOT / "build" / "rocm_tests"))
HIPCC = os.environ.get("HIPCC", "/opt/rocm/bin/hipcc")
ARCH = os.environ.get("HUMMING_ROCM_ARCH", "gfx950")


def _headers_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(INCLUDE_DIR.rglob("*.cuh")):
        digest.update(str(path.relative_to(INCLUDE_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build(source: str, name: str, extra_flags: tuple[str, ...] = ()) -> ctypes.CDLL:
    flags = (
        f"--offload-arch={ARCH}",
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        "-Wno-return-type",
        "-Wno-unused-result",
        f"-I{INCLUDE_DIR}",
        *extra_flags,
        # Experiments: e.g. HUMMING_ROCM_EXTRA_FLAGS="-DHUMMING_MFMA_INTERLEAVE=0".
        *os.environ.get("HUMMING_ROCM_EXTRA_FLAGS", "").split(),
    )
    key_source = "\n".join((source, _headers_digest(), " ".join(flags)))
    key = hashlib.sha256(key_source.encode()).hexdigest()[:16]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    library_path = BUILD_DIR / f"{name}_{key}.so"
    if not library_path.exists():
        source_path = BUILD_DIR / f"{name}_{key}.hip"
        source_path.write_text(source)
        temp_path = library_path.with_suffix(".so.tmp")
        command = [HIPCC, *flags, str(source_path), "-o", str(temp_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"hipcc failed:\n{' '.join(command)}\n{result.stderr}")
        temp_path.rename(library_path)
    return ctypes.CDLL(str(library_path))

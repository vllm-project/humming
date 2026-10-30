"""Platform layer on ROCm: device fields, single HIP runtime, hipcc JIT, launch ordering."""

import ctypes
from pathlib import Path

import torch

from humming.device import DeviceInfo
from humming.jit import hip
from humming.jit.compiler import HIPCCCompiler
from humming.utils import rocm as rocm_utils
from humming.utils.cubin import get_cubin_kernel_names

AMDGPU_ELF_MACHINE = 0xE0

ADD_ONE_SOURCE = """
#include <humming/utils/platform.cuh>

extern "C" __global__ void humming_test_add_one(float* values, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) values[index] += 1.0f;
}

// Reads lane 0 of each 32-lane logical warp: checks the kLogicalWarpSize shuffle contract.
extern "C" __global__ void humming_test_logical_warp_broadcast(int* output) {
  int lane = threadIdx.x;
  output[lane] = __shfl_sync(kFullWarpMask, lane, 0);
}
"""


def _compile_test_module() -> str:
    device_info = DeviceInfo(torch.cuda.current_device())
    return HIPCCCompiler.compile(ADD_ONE_SOURCE, device_info.arch, "")


def test_device_fields():
    device_info = DeviceInfo(torch.cuda.current_device())
    assert device_info.is_rocm
    assert device_info.sm_version == 0
    assert device_info.warp_size == 64
    assert device_info.arch.startswith("gfx")
    assert ":" not in device_info.arch


def test_single_hip_runtime_is_mapped():
    _compile_test_module()
    hip.get_device()
    maps_text = Path("/proc/self/maps").read_text()
    mapped_runtimes = {
        Path(line.split()[-1]).resolve()
        for line in maps_text.splitlines()
        if line.split() and Path(line.split()[-1]).name.startswith("libamdhip64.so")
    }
    assert len(mapped_runtimes) == 1, mapped_runtimes
    assert Path(rocm_utils.get_amdhip_library_path()).resolve() in mapped_runtimes


def test_hipcc_emits_raw_amdgpu_elf_with_kernel_names():
    code_object_path = _compile_test_module()
    header = Path(code_object_path).read_bytes()[:20]
    assert header[:4] == b"\x7fELF"
    elf_machine = int.from_bytes(header[18:20], "little")
    assert elf_machine == AMDGPU_ELF_MACHINE
    kernel_names = set(get_cubin_kernel_names(code_object_path))
    assert {"humming_test_add_one", "humming_test_logical_warp_broadcast"} <= kernel_names


def test_launch_is_ordered_with_torch_stream():
    module = hip.module_load(_compile_test_module())
    function = hip.module_get_function(module, "humming_test_add_one")
    count = 1 << 20
    num_launches = 8
    stream = torch.cuda.current_stream()
    for _ in range(4):
        values = torch.zeros(count, dtype=torch.float32, device="cuda")
        for _ in range(num_launches):
            hip.launch_kernel(
                function,
                ((count + 255) // 256,),
                (256,),
                stream.cuda_stream,
                [values.data_ptr(), count],
                [ctypes.c_void_p, ctypes.c_int],
            )
        # No explicit synchronize: the copy must be ordered after the launches on torch's stream.
        host_values = values.cpu()
        assert torch.all(host_values == num_launches)


def test_logical_warp_shuffle_is_32_lanes():
    module = hip.module_load(_compile_test_module())
    function = hip.module_get_function(module, "humming_test_logical_warp_broadcast")
    output = torch.full((64,), -1, dtype=torch.int32, device="cuda")
    hip.launch_kernel(
        function,
        (1,),
        (64,),
        torch.cuda.current_stream().cuda_stream,
        [output.data_ptr()],
        [ctypes.c_void_p],
    )
    expected = torch.tensor([0] * 32 + [32] * 32, dtype=torch.int32)
    assert torch.equal(output.cpu(), expected)

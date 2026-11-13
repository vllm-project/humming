import os

import cuda.bindings.driver as cbd
import pytest
import torch

from humming import ops
from humming.kernel.dequant_weight import DequantKernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("num_elements", (1, 31, 32, 33, 1023, 1024, 1025, 4096))
def test_dequant_fp4_values(num_elements):
    inputs = torch.arange(num_elements, dtype=torch.int32, device="cuda") % 16
    outputs = torch.full((num_elements,), float("nan"), dtype=torch.float32, device="cuda")
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device="cuda",
    )

    ops.dequant_weight(inputs, 2, 1, True, outputs=outputs)

    torch.testing.assert_close(outputs, values[inputs.long()], rtol=0, atol=0)


@pytest.mark.parametrize("total_size", (1024, 2**32, 2**32 + 17))
def test_dequant_large_count_kernel_argument(total_size):
    # One warp processes 32 values per thread: only the first 1024 elements
    # are accessed. A large logical count tests both sides of the ABI without
    # requiring a 32 GiB allocation in the regular test suite.
    inputs = torch.full((1024,), 2, dtype=torch.int32, device="cuda")
    outputs = torch.full((1024,), float("nan"), dtype=torch.float32, device="cuda")
    kernel = DequantKernel()
    config = cbd.CUlaunchConfig()
    config.gridDimX = config.gridDimY = config.gridDimZ = 1
    config.blockDimX = 32
    config.blockDimY = config.blockDimZ = 1
    config.hStream = torch.cuda.current_stream().cuda_stream
    args = (inputs.data_ptr(), outputs.data_ptr(), total_size, 2, 1, True)

    (result,) = cbd.cuLaunchKernelEx(config, kernel.load_cubin(), (args, kernel.arg_types), 0)
    assert result == cbd.CUresult.CUDA_SUCCESS

    torch.testing.assert_close(outputs, torch.ones_like(outputs), rtol=0, atol=0)


def test_dequant_more_than_uint32_elements():
    num_elements = 2**32 + 1024
    required_bytes = num_elements * 8 + 1024**3
    if torch.cuda.mem_get_info()[0] < required_bytes:
        pytest.skip("not enough free GPU memory for the large-tensor regression")

    inputs = torch.full((num_elements,), 2, dtype=torch.int32, device="cuda")
    outputs = torch.full((num_elements,), float("nan"), dtype=torch.float32, device="cuda")

    ops.dequant_weight(inputs, 2, 1, True, outputs=outputs)

    # Check every value while bounding the size of temporary comparison tensors.
    for chunk in outputs.split(2**26):
        assert torch.all(chunk == 1.0).item()

"""Single-MFMA layout check: verifies the lane -> (row, k) mappings in mma/mfma.cuh on hardware."""

import ctypes

import pytest
import torch

from tests.rocm.hip_build import build

SOURCE = r"""
#include <humming/mma/mfma.cuh>

template <class ElementA, uint32_t kShapeM>
__global__ void single_mfma_kernel(const uint16_t *a_ptr, const uint16_t *b_ptr, float *c_ptr) {
  using Op = MfmaOp<ElementA, kShapeM>;
  const uint32_t lane = threadIdx.x;
  uint16_t regs_a[Op::kElemsPerLane];
  uint16_t regs_b[Op::kElemsPerLane];
  for (uint32_t slot = 0; slot < Op::kElemsPerLane; slot++) {
    // A is row-major [M, K]; B is given as row-major [N, K] (i.e. B^T, like a weight).
    regs_a[slot] = a_ptr[Op::a_row(lane) * Op::K + Op::ab_k(lane, slot)];
    regs_b[slot] = b_ptr[Op::b_col(lane) * Op::K + Op::ab_k(lane, slot)];
  }
  typename Op::CRegisters regs_c = {};
  Op::fma(reinterpret_cast<uint32_t *>(regs_a), reinterpret_cast<uint32_t *>(regs_b), regs_c);
  for (uint32_t reg = 0; reg < Op::kRegsC; reg++) {
    c_ptr[Op::c_row(lane, reg) * Op::N + Op::c_col(lane)] = regs_c[reg];
  }
}

#define DEFINE_LAUNCH(NAME, TYPE, SHAPE_M)                                                    \
  extern "C" int NAME(const void *a, const void *b, void *c, void *stream) {                \
    hipLaunchKernelGGL((single_mfma_kernel<TYPE, SHAPE_M>), dim3(1), dim3(64), 0,           \
                       (hipStream_t)stream, (const uint16_t *)a, (const uint16_t *)b, (float *)c); \
    return (int)hipGetLastError();                                                          \
  }

DEFINE_LAUNCH(launch_f16_16, Float16, 16)
DEFINE_LAUNCH(launch_bf16_16, BFloat16, 16)
DEFINE_LAUNCH(launch_f16_32, Float16, 32)
DEFINE_LAUNCH(launch_bf16_32, BFloat16, 32)
"""

SHAPES = {16: (16, 16, 32), 32: (32, 32, 16)}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape_m", [16, 32])
def test_single_mfma_layout(dtype: torch.dtype, shape_m: int):
    library = build(SOURCE, "mfma_layout")
    m, n, k = SHAPES[shape_m]
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(n, k, dtype=dtype, device="cuda")
    c = torch.full((m, n), float("nan"), dtype=torch.float32, device="cuda")
    prefix = "f16" if dtype == torch.float16 else "bf16"
    launch = getattr(library, f"launch_{prefix}_{shape_m}")
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    status = launch(
        ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(c.data_ptr()), stream
    )
    assert status == 0
    torch.cuda.synchronize()
    reference = a.float() @ b.float().T
    torch.testing.assert_close(c, reference, atol=1e-4, rtol=1e-4)

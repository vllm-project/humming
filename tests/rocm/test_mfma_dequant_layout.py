"""Checks that layout.repack_codes_for_mfma + Humming's dequant produce the MFMA B fragments."""

import ctypes

import pytest
import torch

from humming.rocm import layout
from tests.rocm.hip_build import build

SOURCE_TEMPLATE = r"""
#include <humming/kernel/humming_mfma.cuh>

// One 16 x 128 chunk: lane l writes its 32 dequantized values (MFMA t, slot s) to out[l][t * 8 + s].
__global__ void dequant_chunk_kernel(
    const uint32_t *qb_ptr, const uint8_t *zp_ptr, const uint16_t *scale_ptr, uint16_t *out_ptr) {{
  using Dequant = MfmaDequant<UInt{bits}, {element_a}, {has_zero_point}>;
  const uint32_t lane = threadIdx.x;
  uint32_t qb[{bits}];
  for (uint32_t i = 0; i < {bits}; i++) qb[i] = qb_ptr[lane * {bits} + i];
  uint32_t zero_point = Dequant::prepare_zero_point(zp_ptr[lane % 16]);
  for (uint32_t t = 0; t < 4; t++) {{
    uint32_t regs_b[4];
    if ({scaled}) {{
      Dequant::run_scaled(qb, regs_b, t, zero_point, scale_ptr[lane % 16]);
    }} else {{
      Dequant::run(qb, regs_b, t, zero_point);
    }}
    for (uint32_t i = 0; i < 4; i++) {{
      out_ptr[lane * 32 + t * 8 + i * 2] = regs_b[i] & 0xFFFF;
      out_ptr[lane * 32 + t * 8 + i * 2 + 1] = regs_b[i] >> 16;
    }}
  }}
}}

extern "C" int launch(const void *qb, const void *zp, const void *scale, void *out, void *stream) {{
  hipLaunchKernelGGL(dequant_chunk_kernel, dim3(1), dim3(64), 0, (hipStream_t)stream,
                     (const uint32_t *)qb, (const uint8_t *)zp, (const uint16_t *)scale, (uint16_t *)out);
  return (int)hipGetLastError();
}}
"""


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("has_zero_point", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_dequant_fragment_layout(dtype: torch.dtype, num_bits: int, has_zero_point: bool, scaled: bool):
    """run() is exact; run_scaled() must equal one rounding of (code - zp) * scale (exact in FP32)."""
    source = SOURCE_TEMPLATE.format(
        scaled="true" if scaled else "false",
        bits=num_bits,
        element_a="Float16" if dtype == torch.float16 else "BFloat16",
        has_zero_point="true" if has_zero_point else "false",
    )
    library = build(source, "mfma_dequant_layout")
    codes = torch.randint(0, 1 << num_bits, (16, 128), dtype=torch.int32, device="cuda")
    if has_zero_point:
        zero_point = torch.randint(0, 1 << num_bits, (16,), dtype=torch.int32, device="cuda")
    else:
        zero_point = torch.full((16,), 1 << (num_bits - 1), dtype=torch.int32, device="cuda")
    words = layout.repack_codes_for_mfma(codes, num_bits)
    # Scales spanning several binades, including ones whose products round.
    scale = (
        (torch.rand(16, device="cuda") * 3 + 0.01).to(dtype)
        if scaled
        else torch.ones(16, dtype=dtype, device="cuda")
    )
    outputs = torch.empty(64, 32, dtype=dtype, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    status = library.launch(
        ctypes.c_void_p(words.data_ptr()),
        ctypes.c_void_p(zero_point.to(torch.uint8).data_ptr()),
        ctypes.c_void_p(scale.data_ptr()),
        ctypes.c_void_p(outputs.data_ptr()),
        ctypes.c_void_p(stream),
    )
    assert status == 0
    torch.cuda.synchronize()

    lanes = torch.arange(64, device="cuda")
    values = torch.arange(32, device="cuda")
    rows = (lanes % 16)[:, None].expand(64, 32)
    ks = (values // 8 * 32 + (lanes // 16)[:, None] * 8 + values % 8).expand(64, 32)
    expected = ((codes[rows, ks] - zero_point[rows]).float() * scale[rows].float()).to(dtype)
    torch.testing.assert_close(outputs, expected, atol=0, rtol=0)

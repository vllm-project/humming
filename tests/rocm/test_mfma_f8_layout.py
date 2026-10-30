"""gfx950 low-bit MFMA fragment layouts, the byte LUT and the B repack (humming/rocm/layout_f8.py).

Single-wave probes of the wrappers in mma/mfma_f8f6f4.cuh: every lane loads its operand registers
from the per-lane layout produced by ``layout_f8.repack_codes`` (the same code that lays out the
weights), so a pass verifies the K order, the bit packing and the E8M0 scale operand semantics
on hardware.
"""

import ctypes

import pytest
import torch

from humming import dtypes
from humming.config import LayerConfig
from humming.kernel import humming_mfma_f8 as f8
from humming.rocm import layout_f8
from tests.rocm.hip_build import build

SOURCE = r"""
#include <humming/mma/mfma_f8f6f4.cuh>

// a / b: 64 lanes x 8 words; sa / sb: 64 scale words; c: 16 x 16 row-major.
template <uint32_t kFmtA, uint32_t kFmtB>
__global__ void scaled_kernel(
    const uint32_t *a, const uint32_t *b, const uint32_t *sa, const uint32_t *sb, float *c) {
  const uint32_t lane = threadIdx.x;
  uint32_t regs_a[8], regs_b[8];
  for (uint32_t i = 0; i < 8; i++) {
    regs_a[i] = a[lane * 8 + i];
    regs_b[i] = b[lane * 8 + i];
  }
  mfma_f32x4_t acc = {};
  mfma_scale_16x16x128<kFmtA, kFmtB>(regs_a, regs_b, acc, sa[lane], sb[lane]);
  for (uint32_t r = 0; r < 4; r++) c[((lane / 16) * 4 + r) * 16 + lane % 16] = acc[r];
}

template <uint32_t kRegs>
__global__ void int8_kernel(const uint32_t *a, const uint32_t *b, int32_t *c) {
  const uint32_t lane = threadIdx.x;
  uint32_t regs_a[4], regs_b[4];
  for (uint32_t i = 0; i < kRegs; i++) {
    regs_a[i] = a[lane * kRegs + i];
    regs_b[i] = b[lane * kRegs + i];
  }
  mfma_i32x4_t acc = {};
  if constexpr (kRegs == 4) {
    mfma_i8_16x16x64(regs_a, regs_b, acc);
  } else {
    mfma_i8_16x16x32(regs_a, regs_b, acc);
  }
  for (uint32_t r = 0; r < 4; r++) c[((lane / 16) * 4 + r) * 16 + lane % 16] = acc[r];
}

__global__ void fp8_x32_kernel(const uint32_t *a, const uint32_t *b, float *c) {
  const uint32_t lane = threadIdx.x;
  uint32_t regs_a[2] = {a[lane * 2], a[lane * 2 + 1]};
  uint32_t regs_b[2] = {b[lane * 2], b[lane * 2 + 1]};
  mfma_f32x4_t acc = {};
  mfma_f8_16x16x32<kMfmaFmtE4M3, kMfmaFmtE4M3>(regs_a, regs_b, acc);
  for (uint32_t r = 0; r < 4; r++) c[((lane / 16) * 4 + r) * 16 + lane % 16] = acc[r];
}

template <uint32_t kBits>
__global__ void lut_kernel(const uint32_t *x, const uint32_t *table, uint32_t *out, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t t[8];
  for (uint32_t j = 0; j < 8; j++) t[j] = table[j];
  if (i < n) out[i] = mfma_lut_lookup<kBits>(x[i], t);
}

#define SCALED(NAME, FA, FB)                                                                 \
  extern "C" int NAME(const void *a, const void *b, const void *sa, const void *sb, void *c,  \
                      void *s) {                                                               \
    hipLaunchKernelGGL((scaled_kernel<FA, FB>), dim3(1), dim3(64), 0, (hipStream_t)s,         \
                       (const uint32_t *)a, (const uint32_t *)b, (const uint32_t *)sa,        \
                       (const uint32_t *)sb, (float *)c);                                      \
    return (int)hipGetLastError();                                                             \
  }
SCALED(scaled_0_0, 0, 0)
SCALED(scaled_1_0, 1, 0)
SCALED(scaled_0_1, 0, 1)
SCALED(scaled_0_2, 0, 2)
SCALED(scaled_0_3, 0, 3)
SCALED(scaled_0_4, 0, 4)
SCALED(scaled_4_0, 4, 0)
SCALED(scaled_4_4, 4, 4)
SCALED(scaled_4_2, 4, 2)
SCALED(scaled_2_3, 2, 3)

extern "C" int int8_x64(const void *a, const void *b, void *c, void *s) {
  hipLaunchKernelGGL((int8_kernel<4>), dim3(1), dim3(64), 0, (hipStream_t)s,
                     (const uint32_t *)a, (const uint32_t *)b, (int32_t *)c);
  return (int)hipGetLastError();
}
extern "C" int int8_x32(const void *a, const void *b, void *c, void *s) {
  hipLaunchKernelGGL((int8_kernel<2>), dim3(1), dim3(64), 0, (hipStream_t)s,
                     (const uint32_t *)a, (const uint32_t *)b, (int32_t *)c);
  return (int)hipGetLastError();
}
extern "C" int fp8_x32(const void *a, const void *b, void *c, void *s) {
  hipLaunchKernelGGL(fp8_x32_kernel, dim3(1), dim3(64), 0, (hipStream_t)s,
                     (const uint32_t *)a, (const uint32_t *)b, (float *)c);
  return (int)hipGetLastError();
}
#define LUT(BITS)                                                                                  \
  extern "C" int lut_##BITS(const void *x, const void *t, void *o, unsigned n, void *s) {          \
    hipLaunchKernelGGL((lut_kernel<BITS>), dim3((n + 255) / 256), dim3(256), 0, (hipStream_t)s,    \
                       (const uint32_t *)x, (const uint32_t *)t, (uint32_t *)o, n);                \
    return (int)hipGetLastError();                                                                 \
  }
LUT(3)
LUT(4)
LUT(5)
"""

# MFMAs with an 8-bit operand do not accumulate in full FP32: measured max |error| / sum |a b|
# on MI355X is ~1-2e-4 (FP4 / FP6 x FP4 / FP6: ~2e-8). A wrong K order or scale mapping gives
# ~1 / sqrt(K) ~ 0.1.
ACCUMULATION_RTOL = 1e-3

# format code -> (bits, exponent bits, mantissa bits)
FORMATS = {0: (8, 4, 3), 1: (8, 5, 2), 2: (6, 2, 3), 3: (6, 3, 2), 4: (4, 2, 1)}


def _library():
    return build(SOURCE, "mfma_f8_layout")


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def decode_fp(codes: torch.Tensor, exponent_bits: int, mantissa_bits: int) -> torch.Tensor:
    """OCP FP decode (no inf / NaN handling; callers avoid those codes)."""
    codes = codes.to(torch.int64)
    sign = 1 - 2 * ((codes >> (exponent_bits + mantissa_bits)) & 1).float()
    exponent = (codes >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = (codes & ((1 << mantissa_bits) - 1)).float() / (1 << mantissa_bits)
    bias = (1 << (exponent_bits - 1)) - 1
    normal = (1 + mantissa) * torch.exp2((exponent - bias).float())
    subnormal = mantissa * 2.0 ** (1 - bias)
    return sign * torch.where(exponent == 0, subnormal, normal)


def assert_accumulation_close(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """FP32-accumulated C vs the exact product, bounded by K * eps * sum |a| |b| (a layout error is O(1))."""
    a, b = a.double(), b.double()
    reference = a @ b.T
    magnitude = a.abs() @ b.abs().T + 1e-30
    error = (c.cpu().double() - reference).abs()
    assert (error <= ACCUMULATION_RTOL * magnitude).all(), (
        f"max error / sum|ab| {(error / magnitude).max().item():.3g}"
    )


def random_codes(fmt: int, shape, generator) -> torch.Tensor:
    bits, exponent_bits, mantissa_bits = FORMATS[fmt]
    codes = torch.randint(0, 1 << bits, shape, generator=generator)
    if bits == 8:  # avoid the e4m3 NaN codes and the e5m2 inf / NaN codes
        max_exponent = (1 << exponent_bits) - 1
        exponent = (codes >> mantissa_bits) & max_exponent
        codes = torch.where(
            exponent == max_exponent, codes & ~(1 << (mantissa_bits + exponent_bits - 1)), codes
        )
    return codes


def lane_registers(codes: torch.Tensor, bits: int, layout: str, regs: int = 8) -> torch.Tensor:
    """[16, 128] codes -> [64 lanes, regs] int32 words in the kernel's per-lane operand layout."""
    words = layout_f8.repack_codes(codes.to(torch.int32), bits, layout, "stream").reshape(64, bits)
    if regs > bits:
        words = torch.cat([words, torch.zeros(64, regs - bits, dtype=torch.int32)], dim=1)
    return words.contiguous()


@pytest.mark.parametrize(
    ("fmt_a", "fmt_b"), [(0, 0), (1, 0), (0, 1), (0, 2), (0, 3), (0, 4), (4, 0), (4, 4), (4, 2), (2, 3)]
)
def test_scaled_mfma_layout(fmt_a: int, fmt_b: int):
    library = _library()
    generator = torch.Generator().manual_seed(fmt_a * 10 + fmt_b)
    codes_a = random_codes(fmt_a, (16, 128), generator)
    codes_b = random_codes(fmt_b, (16, 128), generator)
    layout_a = "L128" if FORMATS[fmt_a][0] == 8 else "TRUE"
    layout_b = "L128" if FORMATS[fmt_b][0] == 8 else "TRUE"
    regs_a = lane_registers(codes_a, FORMATS[fmt_a][0], layout_a)
    regs_b = lane_registers(codes_b, FORMATS[fmt_b][0], layout_b)
    # E8M0 scales in byte 0; garbage above it checks that only byte 0 is read (opsel 0).
    exponents_a = torch.randint(122, 133, (64,), generator=generator)
    exponents_b = torch.randint(122, 133, (64,), generator=generator)
    scale_a = (exponents_a | (torch.randint(0, 1 << 23, (64,), generator=generator) << 8)).to(torch.int32)
    scale_b = (exponents_b | (torch.randint(0, 1 << 23, (64,), generator=generator) << 8)).to(torch.int32)

    c = torch.full((16, 16), float("nan"), dtype=torch.float32, device="cuda")
    args = [t.cuda() for t in (regs_a, regs_b, scale_a, scale_b)]
    status = getattr(library, f"scaled_{fmt_a}_{fmt_b}")(*map(_ptr, args), _ptr(c), _stream())
    assert status == 0
    torch.cuda.synchronize()

    # Lane l scales row / column l % 16 for the K block [32 (l / 16), 32 (l / 16) + 32).
    k_block = torch.arange(128) // 32
    row = torch.arange(16).view(-1, 1)
    factor_a = torch.exp2((exponents_a[k_block.view(1, -1) * 16 + row] - 127).float())
    factor_b = torch.exp2((exponents_b[k_block.view(1, -1) * 16 + row] - 127).float())
    a = decode_fp(codes_a, *FORMATS[fmt_a][1:]) * factor_a
    b = decode_fp(codes_b, *FORMATS[fmt_b][1:]) * factor_b
    assert_accumulation_close(c, a, b)


@pytest.mark.parametrize("shape_k", [64, 32])
def test_int8_mfma_layout(shape_k: int):
    library = _library()
    generator = torch.Generator().manual_seed(shape_k)
    codes_a = torch.randint(-128, 128, (16, 128), generator=generator)
    codes_b = torch.randint(-128, 128, (16, 128), generator=generator)
    layout, regs = ("L128", 4) if shape_k == 64 else ("L32", 2)
    # The first `regs` words of a lane hold the K slice [0, shape_k) of both layouts.
    regs_a = lane_registers(codes_a & 0xFF, 8, layout)[:, :regs].contiguous()
    regs_b = lane_registers(codes_b & 0xFF, 8, layout)[:, :regs].contiguous()
    c = torch.zeros(16, 16, dtype=torch.int32, device="cuda")
    regs_a, regs_b = regs_a.cuda(), regs_b.cuda()  # keep the device copies alive across the launch
    status = getattr(library, f"int8_x{shape_k}")(_ptr(regs_a), _ptr(regs_b), _ptr(c), _stream())
    assert status == 0
    torch.cuda.synchronize()
    reference = codes_a[:, :shape_k] @ codes_b[:, :shape_k].T
    assert torch.equal(c.cpu().to(torch.int64), reference)


def test_fp8_x32_mfma_layout():
    library = _library()
    generator = torch.Generator().manual_seed(7)
    codes_a = random_codes(0, (16, 128), generator)
    codes_b = random_codes(0, (16, 128), generator)
    regs_a = lane_registers(codes_a, 8, "L32")[:, :2].contiguous()
    regs_b = lane_registers(codes_b, 8, "L32")[:, :2].contiguous()
    c = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    regs_a, regs_b = regs_a.cuda(), regs_b.cuda()
    assert library.fp8_x32(_ptr(regs_a), _ptr(regs_b), _ptr(c), _stream()) == 0
    torch.cuda.synchronize()
    a = decode_fp(codes_a[:, :32], 4, 3).double()
    b = decode_fp(codes_b[:, :32], 4, 3).double()
    assert_accumulation_close(c, a, b)


@pytest.mark.parametrize("index_bits", [3, 4, 5])
def test_lut_lookup(index_bits: int):
    library = _library()
    generator = torch.Generator().manual_seed(index_bits)
    entries = 1 << index_bits
    table = torch.randint(0, 256, (entries,), generator=generator)
    words = torch.tensor(layout_f8.table_words(table), dtype=torch.int64)
    words = torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)
    indices = torch.randint(0, entries, (4096, 4), generator=generator)
    x = indices[:, 0] | (indices[:, 1] << 8) | (indices[:, 2] << 16) | (indices[:, 3] << 24)
    x = torch.where(x >= 1 << 31, x - (1 << 32), x).to(torch.int32).cuda()
    out = torch.zeros(4096, dtype=torch.int32, device="cuda")
    words = words.cuda()
    status = getattr(library, f"lut_{index_bits}")(_ptr(x), _ptr(words), _ptr(out), 4096, _stream())
    assert status == 0
    torch.cuda.synchronize()
    out = out.cpu().to(torch.int64) & 0xFFFFFFFF
    got = torch.stack([(out >> (8 * i)) & 0xFF for i in range(4)], dim=1)
    assert torch.equal(got, table[indices])


@pytest.mark.parametrize("layout", layout_f8.LAYOUTS)
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("packing", ["stream", "planes"])
def test_repack_round_trip(layout: str, bits: int, packing: str):
    generator = torch.Generator().manual_seed(bits)
    codes = torch.randint(0, 1 << bits, (64, 512), generator=generator, dtype=torch.int32)
    weight = layout_f8.repack_codes(codes, bits, layout, packing)
    assert weight.shape == (64, 512 * bits // 32) and weight.dtype == torch.int32
    assert torch.equal(layout_f8.unrepack_weight(weight, bits, layout, packing), codes)


def test_lane_k_index_is_a_permutation():
    for layout in layout_f8.LAYOUTS:
        k_index = layout_f8.lane_k_index(layout)
        for column in range(16):  # the 4 lanes of a column (16 apart) cover K = 0..127 exactly once
            ks = k_index[torch.arange(4) * 16 + column]
            assert sorted(ks.flatten().tolist()) == list(range(128))


@pytest.mark.parametrize(
    ("bits", "has_zero_point"), [(b, z) for b in range(1, 6) for z in (False, True) if b + z <= 5]
)
def test_int_table_exact_in_e4m3(bits: int, has_zero_point: bool):
    values = layout_f8.int_table_values(bits, has_zero_point)
    encoded = layout_f8.table_bytes(values, "fp8")
    assert encoded is not None
    decoded = torch.tensor(encoded.tolist(), dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    assert torch.equal(decoded, values)


def test_e8m0_and_planes_helpers():
    scales = torch.tensor([[120, 127], [130, 0]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert torch.equal(
        layout_f8.group_scale_e8m0(scales), torch.tensor([[120, 130], [127, 0]], dtype=torch.uint8)
    )
    assert layout_f8.plane_widths(5) == [4, 1] and layout_f8.plane_widths(7) == [4, 2, 1]
    assert layout_f8.plane_widths(8) == [8]


def _config(a_dtype, b_dtype, **kwargs) -> LayerConfig:
    return LayerConfig(
        shape_n=1024, shape_k=2048, a_dtype=a_dtype, b_dtype=b_dtype, c_dtype=dtypes.bfloat16, **kwargs
    )


def test_plan_selection():
    plan = f8.plan_mfma_f8(_config(dtypes.float8e4m3, dtypes.float8e4m3, bs_dtype=dtypes.float32))
    assert plan.b_mode == f8.B_NATIVE and plan.mfma_k == 128 and plan.layout == "L128"
    plan = f8.plan_mfma_f8(
        _config(dtypes.float8e4m3, dtypes.int4, bs_dtype=dtypes.bfloat16, weight_scale_group_size=128)
    )
    assert plan.b_mode == f8.B_LUT and plan.mfma_k == 128 and plan.ws_mode == f8.SCALE_FLUSH
    plan = f8.plan_mfma_f8(
        _config(dtypes.float8e4m3, dtypes.float4e2m1, bs_dtype=dtypes.float8e8m0, weight_scale_group_size=32)
    )
    assert plan.b_mode == f8.B_NATIVE and plan.layout == "TRUE" and plan.ws_mode == f8.SCALE_HW
    plan = f8.plan_mfma_f8(
        _config(dtypes.int8, dtypes.int4, bs_dtype=dtypes.bfloat16, weight_scale_group_size=128)
    )
    assert plan.b_mode == f8.B_INT_ARITH and plan.mfma_k == 64

"""ROCm gfx950 low-bit activation GEMM (``humming/kernel/humming_mfma_f8.cuh``) behind Humming's API.

Layers whose activation is FP8 (e4m3 / e5m2), INT8 or MXFP4 (e2m1) are planned here: ``plan_mfma_f8``
maps a ``LayerConfig`` to the kernel's compile-time modes (MFMA shape, B handling, scale modes)
or to the reason it is not supported, which ``check_mfma_layer_config`` raises as
``NotImplementedError``. ``transform_mfma_f8_tensors`` builds the kernel's tensors and
``HummingMfmaF8Kernel`` JIT-compiles one tile; the C++ launcher recognises the kernel by its
``MFMA_F8_MODE`` symbol (csrc/launcher/mfma_f8.h).

Supported (see docs/rocm/log_f8.md for the full matrix):

* FP8 e4m3 / e5m2 activations (per-tensor / per-token FP32 scales, per-group FP32 scales with a
  group multiple of 32, MX E8M0 group-32 scales) x
  - FP8 e4m3 / e5m2, FP6 e2m3 / e3m2, FP4 e2m1 weights fed natively to the MFMA,
  - INT1-5 weights (symmetric), INT1-4 weights with zero points, and other FP formats whose
    values are exact in e4m3 (e.g. FP4 e2m1 when FP32 group scales force K = 32 MFMAs), via a
    table lookup to e4m3;
* INT8 activations x INT1-8 weights (zero points up to 7 bits) and MXFP4 weights;
* MXFP4 activations (E8M0 group-32 scales, or FP32 per-token / per-tensor) x FP4 / FP6 / FP8
  weights natively and INT / small-FP weights through the e4m3 table;

with per-tensor / per-channel / group (FP32-like or E8M0) / block weight scales and per-tensor /
per-channel secondary weight scales.
"""

import dataclasses
import json
import math
from typing import ClassVar

import jinja2
import torch

from humming import dtypes
from humming.config import LayerConfig
from humming.config.enum import InputQuantizationMode, WeightScale2Type, WeightScaleType
from humming.jit.runtime import KernelRuntime
from humming.tune.gfx950 import MAX_SPLITS, MfmaTile

CHUNK_K = 128
INT8_MAX_K = 131072  # |a * b| <= 2^14, so INT32 partial sums are exact up to K = 2^17

# MFMA operand format codes (mma/mfma_f8f6f4.cuh).
FMT_E4M3, FMT_E5M2, FMT_E2M3, FMT_E3M2, FMT_E2M1, FMT_INT8 = 0, 1, 2, 3, 4, 8
B_NATIVE, B_LUT, B_INT_ARITH = 0, 1, 2
SCALE_NONE, SCALE_FLUSH, SCALE_HW, SCALE_E8M0_FLUSH = 0, 1, 2, 3

FP_FORMATS = {
    dtypes.float8e4m3: FMT_E4M3,
    dtypes.float8e5m2: FMT_E5M2,
    dtypes.float6e2m3: FMT_E2M3,
    dtypes.float6e3m2: FMT_E3M2,
    dtypes.float4e2m1: FMT_E2M1,
}


@dataclasses.dataclass(frozen=True)
class MfmaF8Plan:
    a_fmt: int
    b_fmt: int
    b_mode: int
    bits: int
    lut_bits: int
    lut: tuple[int, ...]
    has_zero_point: bool
    mfma_k: int
    scale_k: int
    ws_mode: int
    ws_group: int
    zp_group: int
    as_mode: int
    as_group: int
    layout: str  # B lane value order (layout_f8.py)
    packing: str  # "stream" (native operand) or "planes"
    bs2_factor: float  # folded into the FP32 [N] epilogue vector (INT8 x FP4 table doubling)

    @property
    def a_pack(self) -> int:
        """K values per stored activation byte."""
        return 2 if self.a_fmt == FMT_E2M1 else 1


class _Unsupported(Exception):
    pass


def _effective_weight_group(config: LayerConfig) -> int:
    """Weight scale K group (0 = one scale over K: tensor / channel / single group)."""
    if config.weight_scale_type in (WeightScaleType.TENSOR, WeightScaleType.CHANNEL):
        return 0
    group = config.weight_scale_group_size
    return 0 if group >= config.shape_k else group


def _a_format(config: LayerConfig) -> int:
    a_dtype = config.a_dtype
    if a_dtype in (dtypes.float8e4m3, dtypes.float8e5m2):
        return FP_FORMATS[a_dtype]
    if a_dtype == dtypes.float4e2m1:
        return FMT_E2M1
    if a_dtype == dtypes.int8:
        return FMT_INT8
    raise _Unsupported(
        f"activation dtype {a_dtype} (low-bit activations: float8e4m3, float8e5m2, int8, float4e2m1)"
    )


def _activation_scale(config: LayerConfig) -> tuple[str, int]:
    """("tensor" | "token" | "fp32" | "e8m0", group size) of the activation scale."""
    mode = InputQuantizationMode(config.input_quant_mode)
    if mode == InputQuantizationMode.StaticTensor:
        return "tensor", 0
    if mode == InputQuantizationMode.DynamicToken:
        return "token", 0
    if mode == InputQuantizationMode.DynamicGroup:
        group = config.input_scale_group_size
        if config.pad_shape_k:
            raise _Unsupported(
                f"per-group activation scales with K={config.shape_k - config.pad_shape_k} "
                f"not a multiple of {CHUNK_K}"
            )
        if group % 32 or config.shape_k % group:
            raise _Unsupported(f"activation scale group size {group} (must be a multiple of 32 dividing K)")
        if group >= config.shape_k:
            raise _Unsupported("a single activation scale group spanning K (use per-token scales)")
        if config.as_dtype == dtypes.float32:
            return "fp32", group
        if config.as_dtype == dtypes.float8e8m0:
            return "e8m0", group
        raise _Unsupported(f"activation group scale dtype {config.as_dtype} (float32 or float8e8m0)")
    raise _Unsupported(f"input quantization mode {mode.value}")


def _weight_scale(config: LayerConfig) -> tuple[str, int]:
    """("fold" | "fp32" | "e8m0", K group) of the primary weight scale."""
    group = _effective_weight_group(config)
    if config.weight_scale_type == WeightScaleType.BLOCK:
        if group == 0:
            return "fold_block", 0
        return "fp32", group
    if group == 0:
        if config.bs_dtype == dtypes.float8e8m0:
            return "fold_e8m0", 0
        return "fold", 0
    if group % 32 or config.shape_k % group:
        raise _Unsupported(f"weight scale group size {group} (must be a multiple of 32 dividing K)")
    if config.bs_dtype == dtypes.float8e8m0:
        return "e8m0", group
    if config.bs_dtype == dtypes.float8e4m3 and group == 16:
        raise _Unsupported("NVFP4-style group-16 e4m3 weight scales")
    return "fp32", group


def _scale_k(groups: list[int]) -> int:
    groups = [g for g in groups if g]
    if not groups:
        return 0
    scale_k = functools_gcd(groups)
    if scale_k < CHUNK_K and CHUNK_K % scale_k:
        raise _Unsupported(f"scale group sizes {groups} (their gcd must divide or be a multiple of 128)")
    if scale_k >= CHUNK_K and scale_k % CHUNK_K:
        raise _Unsupported(f"scale group sizes {groups} (their gcd must divide or be a multiple of 128)")
    return scale_k


def functools_gcd(values: list[int]) -> int:
    result = 0
    for value in values:
        result = math.gcd(result, value)
    return result


def _table(values: torch.Tensor, target: str, what: str) -> tuple[int, ...]:
    from humming.rocm import layout_f8

    encoded = layout_f8.table_bytes(values, target)
    if encoded is None:
        raise _Unsupported(
            f"{what} (values not exactly representable as {'e4m3' if target == 'fp8' else 'int8'})"
        )
    return layout_f8.table_words(encoded)


def _plan(config: LayerConfig) -> MfmaF8Plan:
    if config.num_experts:
        raise _Unsupported("MoE layers (num_experts > 0)")
    a_fmt = _a_format(config)
    is_int8 = a_fmt == FMT_INT8
    is_fp4_a = a_fmt == FMT_E2M1
    b_dtype = config.b_dtype
    bits = b_dtype.num_bits
    has_zero_point = config.has_zero_point
    if has_zero_point and config.is_fp_zero_point:
        raise _Unsupported("floating-point zero points")
    if config.shape_n % 64 or config.shape_k % CHUNK_K:
        raise _Unsupported(f"padded shape N={config.shape_n}, K={config.shape_k}")
    if config.c_dtype not in (dtypes.float16, dtypes.bfloat16):
        raise _Unsupported(f"output dtype {config.c_dtype}")

    as_kind, as_group = _activation_scale(config)
    ws_kind, ws_group = _weight_scale(config)
    if is_fp4_a and as_kind == "fp32":
        raise _Unsupported("FP4 activations with FP32 group scales (only E8M0 group-32 or per-token scales)")

    # B handling: native MFMA operand, e4m3 / int8 table, or int8 arithmetic.
    b_is_int = b_dtype.is_integer_type
    native_fmt = None
    if not b_is_int and b_dtype in FP_FORMATS and not has_zero_point:
        if is_int8:
            native_fmt = None
        elif b_dtype == dtypes.float4e2m1 or b_dtype in (dtypes.float6e2m3, dtypes.float6e3m2):
            native_fmt = FP_FORMATS[b_dtype]
        else:
            native_fmt = FP_FORMATS[b_dtype]  # fp8 x fp8 (e4m3 / e5m2)
    if b_is_int and has_zero_point and bits == 8:
        raise _Unsupported("8-bit integer weights with zero points")

    # Accumulator flush granularity: FP32-like scales (and E8M0 scales the MFMA cannot apply).
    def flush_groups(hw_ok: bool) -> list[int]:
        groups = []
        if as_kind == "fp32" or (as_kind == "e8m0" and not hw_ok):
            groups.append(as_group)
        if ws_kind == "fp32" or (ws_kind == "e8m0" and not hw_ok):
            groups.append(ws_group)
        return groups

    if is_int8:
        scale_k = _scale_k(flush_groups(False))
        mfma_k = 64 if scale_k == 0 or scale_k % 64 == 0 else 32
        hw_ok = False
    else:
        scale_k = _scale_k(flush_groups(True))
        mfma_k = 128 if scale_k == 0 or scale_k % CHUNK_K == 0 else 32
        hw_ok = mfma_k == 128
        if not hw_ok:
            if is_fp4_a:
                raise _Unsupported("FP4 activations with scale groups below 128 that need FP32 flushes")
            scale_k = _scale_k(flush_groups(False))
            mfma_k = 32
    if mfma_k == 32 and native_fmt in (FMT_E2M1, FMT_E2M3, FMT_E3M2):
        native_fmt = None  # K = 32 MFMAs take 8-bit operands only: dequantize FP4 / FP6 via the table

    lut = (0,) * 8
    lut_bits = 1
    bs2_factor = 1.0
    if native_fmt is not None:
        b_mode, b_fmt = B_NATIVE, native_fmt
    elif is_int8:
        b_fmt = FMT_INT8
        if b_is_int:
            if bits > 8:
                raise _Unsupported(f"weight dtype {b_dtype} with int8 activations")
            b_mode = B_INT_ARITH
        elif b_dtype == dtypes.float4e2m1 and not has_zero_point:
            from humming.rocm import layout_f8

            # Doubled e2m1 values are integers (0, 1, 2, 3, 4, 6, 8, 12); the 0.5 goes to the epilogue.
            values = layout_f8.fp_table_values(b_dtype.exponent_bits, b_dtype.mantissa_bits, bits) * 2
            lut, lut_bits, b_mode, bs2_factor = (
                _table(values, "int8", f"weight dtype {b_dtype}"),
                bits,
                B_LUT,
                0.5,
            )
        else:
            raise _Unsupported(f"weight dtype {b_dtype} with int8 activations (int1-8 or float4e2m1)")
    else:
        from humming.rocm import layout_f8

        b_fmt, b_mode = FMT_E4M3, B_LUT
        if b_is_int:
            lut_bits = bits + 1 if has_zero_point else bits
            zero_points = " with zero points" if has_zero_point else ""
            what = f"weight dtype {b_dtype}{zero_points} with FP8/FP4 activations"
            if lut_bits > 5:
                raise _Unsupported(f"{what} (int1-5 symmetric, int1-4 with zero points)")
            lut = _table(layout_f8.int_table_values(bits, has_zero_point), "fp8", what)
        else:
            what = f"weight dtype {b_dtype} with FP8/FP4 activations"
            if has_zero_point:
                raise _Unsupported(f"{what} and zero points")
            if bits > 5:
                raise _Unsupported(f"{what} (only native fp8/fp6/fp4 or <= 5-bit formats exact in e4m3)")
            lut_bits = bits
            values = layout_f8.fp_table_values(b_dtype.exponent_bits, b_dtype.mantissa_bits, bits)
            lut = _table(values, "fp8", what)

    if is_int8 and config.shape_k > INT8_MAX_K:
        raise _Unsupported(
            f"int8 activations with K={config.shape_k} > {INT8_MAX_K} (INT32 accumulator range)"
        )

    def scale_mode(kind: str) -> int:
        if kind == "fp32":
            return SCALE_FLUSH
        if kind == "e8m0":
            return SCALE_HW if hw_ok else SCALE_E8M0_FLUSH
        return SCALE_NONE

    ws_mode = scale_mode(ws_kind)
    as_mode = scale_mode(as_kind)
    if b_mode == B_NATIVE and b_fmt in (FMT_E2M1, FMT_E2M3, FMT_E3M2):
        layout, packing = "TRUE", "stream"
    else:
        layout = "L32" if mfma_k == 32 else "L128"
        packing = "stream" if b_mode == B_NATIVE else "planes"
    zp_group = 0
    if has_zero_point:
        zp_group = _effective_weight_group(config)
    return MfmaF8Plan(
        a_fmt=a_fmt,
        b_fmt=b_fmt,
        b_mode=b_mode,
        bits=bits,
        lut_bits=lut_bits,
        lut=lut,
        has_zero_point=has_zero_point,
        mfma_k=mfma_k,
        scale_k=scale_k,
        ws_mode=ws_mode,
        ws_group=ws_group,
        zp_group=zp_group,
        as_mode=as_mode,
        as_group=as_group,
        layout=layout,
        packing=packing,
        bs2_factor=bs2_factor,
    )


_PLAN_CACHE: dict[str, MfmaF8Plan | str] = {}


def plan_mfma_f8(config: LayerConfig) -> MfmaF8Plan | str:
    """Kernel plan of a low-bit activation layer, or the reason it is unsupported."""
    key = config.to_str()
    if key not in _PLAN_CACHE:
        try:
            _PLAN_CACHE[key] = _plan(config)
        except _Unsupported as error:
            _PLAN_CACHE[key] = str(error)
    return _PLAN_CACHE[key]


def is_low_bit_activation(config: LayerConfig) -> bool:
    return config.a_dtype.num_bits < 16


def get_mfma_f8_unsupported_reason(config: LayerConfig) -> str | None:
    plan = plan_mfma_f8(config)
    return plan if isinstance(plan, str) else None


# ----------------------------------------------------------------------------------------------
# Tensor transform.


def transform_mfma_f8_tensors(
    config: LayerConfig, tensors: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Padded Humming tensors -> the low-bit activation kernel's tensors.

    weight [N, K * bits / 32] int32 (layout_f8.py), weight_scale: FP32 [Gw, N] (flushed group /
    block scales), uint8 [Gw, N] (E8M0) or a dummy FP32 [1] when folded, zero_point [Gz, N] uint8,
    weight_scale_2: FP32 [N] epilogue vector (channel x tensor x secondary scales), bias [N].
    """
    from humming.rocm import layout_f8

    plan = plan_mfma_f8(config)
    if isinstance(plan, str):
        raise NotImplementedError(f"Humming on ROCm (gfx950 MFMA GEMM) does not support {plan}.")
    shape_n = config.shape_n
    weight = tensors["weight"]
    device = weight.device
    outputs = {"weight": layout_f8.repack_weight(weight, plan.bits, plan.layout, plan.packing)}

    bs2 = torch.full((shape_n,), plan.bs2_factor, dtype=torch.float32, device=device)
    weight_scale = tensors.get("weight_scale")
    if config.weight_scale_type == WeightScaleType.TENSOR:
        bs2 = bs2 * weight_scale.float().view(-1)[0]
        group_scale = None
    elif config.weight_scale_type == WeightScaleType.BLOCK:
        group_size_n = config.weight_scale_group_size_n
        if plan.ws_mode == SCALE_NONE:
            bs2 = bs2 * weight_scale.float().repeat_interleave(group_size_n, dim=0)[:shape_n, 0]
            group_scale = None
        else:
            group_scale = layout_f8.block_scale_fp32(weight_scale, group_size_n, shape_n)
    elif plan.ws_mode == SCALE_NONE:
        bs2 = bs2 * _decode_scale(config, weight_scale).float().reshape(shape_n, -1)[:, 0]
        group_scale = None
    elif plan.ws_mode == SCALE_FLUSH:
        group_scale = layout_f8.group_scale_fp32(_decode_scale(config, weight_scale))
    else:
        group_scale = layout_f8.group_scale_e8m0(weight_scale)
    outputs["weight_scale"] = (
        group_scale if group_scale is not None else torch.ones(1, dtype=torch.float32, device=device)
    )

    weight_scale_2 = tensors.get("weight_scale_2")
    if weight_scale_2 is not None and config.weight_scale_2_type != WeightScale2Type.NONE:
        ws2 = weight_scale_2.float().view(-1)
        bs2 = bs2 * (ws2[0] if ws2.numel() == 1 else ws2[:shape_n])
    outputs["weight_scale_2"] = bs2.contiguous()

    if config.has_zero_point:
        outputs["zero_point"] = layout_f8.zero_point_codes(tensors["zero_point"], plan.bits)
    if config.has_bias:
        outputs["bias"] = tensors["bias"].to(dtypes.torch_dtype_map[config.c_dtype]).contiguous()
    return outputs


def _decode_scale(config: LayerConfig, weight_scale: torch.Tensor) -> torch.Tensor:
    if config.bs_dtype == dtypes.float8e5m3 and weight_scale.dtype == torch.uint8:
        from humming.utils.weight import decode_e5m3_scale

        return decode_e5m3_scale(weight_scale)
    if weight_scale.dtype == torch.float8_e8m0fnu:
        return weight_scale.float()
    return weight_scale


# ----------------------------------------------------------------------------------------------
# Kernel.

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/humming_mfma_f8.cuh>

using HummingMfmaF8Tile = MfmaF8TileConfig<
    {{tile.warps_m}}, {{tile.warps_n}}, {{tile.tiles_m}}, {{tile.tiles_n}}, {{tile.stages}},
    {{tile.lds_a | lower}}>;

struct HummingMfmaF8Traits {
  using ElementC = {{element_c}};
  static constexpr uint32_t kAFmt = {{p.a_fmt}};
  static constexpr uint32_t kBFmt = {{p.b_fmt}};
  static constexpr uint32_t kBMode = {{p.b_mode}};
  static constexpr uint32_t kBits = {{p.bits}};
  static constexpr uint32_t kLutBits = {{p.lut_bits}};
  static constexpr bool kHasZeroPoint = {{p.has_zero_point | lower}};
  static constexpr uint32_t kMfmaK = {{p.mfma_k}};
  static constexpr uint32_t kScaleK = {{p.scale_k}};
  static constexpr uint32_t kWsMode = {{p.ws_mode}};
  static constexpr uint32_t kWsGroup = {{p.ws_group or 1}};
  static constexpr uint32_t kZpGroup = {{p.zp_group}};
  static constexpr uint32_t kAsMode = {{p.as_mode}};
  static constexpr uint32_t kAsGroup = {{p.as_group or 1}};
  static constexpr uint32_t lut(uint32_t i) {
    return {% for word in p.lut %}i == {{loop.index0}} ? {{word}}u : {% endfor %}0u;
  }
};

// Read by the C++ launcher (csrc/launcher/mfma.h, mfma_f8.h).
extern "C" __constant__ uint32_t MFMA_BLOCK_M = HummingMfmaF8Tile::kBlockM;
extern "C" __constant__ uint32_t MFMA_BLOCK_N = HummingMfmaF8Tile::kBlockN;
extern "C" __constant__ uint32_t MFMA_NUM_THREADS = HummingMfmaF8Tile::kNumThreads;
extern "C" __constant__ uint32_t MFMA_A_DTYPE_ID = {{a_dtype_id}};
extern "C" __constant__ uint32_t MFMA_NUM_BITS = {{p.bits}};
extern "C" __constant__ uint32_t MFMA_GROUP_SIZE = {{p.ws_group}};
extern "C" __constant__ uint32_t MFMA_HAS_ZERO_POINT = {{p.has_zero_point | int}};
extern "C" __constant__ uint32_t MFMA_MAX_SPLITS = {{max_splits}};
extern "C" __constant__ uint32_t MFMA_F8_MODE = 1;
extern "C" __constant__ uint32_t MFMA_F8_C_DTYPE_ID = {{c_dtype_id}};
extern "C" __constant__ uint32_t MFMA_F8_A_PACK = {{p.a_pack}};
extern "C" __constant__ uint32_t MFMA_F8_IS_INT8 = {{(p.a_fmt == 8) | int}};
extern "C" __constant__ uint32_t MFMA_F8_WS_MODE = {{p.ws_mode}};
extern "C" __constant__ uint32_t MFMA_F8_WS_GROUP = {{p.ws_group}};
extern "C" __constant__ uint32_t MFMA_F8_ZP_GROUP = {{p.zp_group}};
extern "C" __constant__ uint32_t MFMA_F8_AS_MODE = {{p.as_mode}};
extern "C" __constant__ uint32_t MFMA_F8_AS_GROUP = {{p.as_group}};
""")


def select_f8_tile(shape_m: int, shape_n: int, num_cus: int, plan: MfmaF8Plan) -> MfmaTile:
    """Tile per M bucket (measured on MI355X, docs/rocm/log_f8.md, "Tiles").

    At low occupancy the mainloop is latency-bound, so small M / N prefer narrow tiles (one 16-col
    MFMA tile per wave, more blocks) with three prefetch stages; 128-row tiles only pay off with a
    few blocks per CU. FP4 activations prefer 2 x 2 waves of 64 x 64 at large M (halves the LDS A
    reads per MFMA). LUT-dequantized weights keep 1 x 4 waves of 128 x 32 (each dequantized B tile
    feeds 8 MFMAs) and two stages (registers).
    """
    fp4_a = plan.a_fmt == FMT_E2M1
    if shape_m <= 8:
        tiles_n = 2 if shape_n // 128 >= num_cus // 2 else 1
        return MfmaTile(tiles_m=1, tiles_n=tiles_n, stages=4)
    if shape_m <= 32:
        tile = (
            MfmaTile(tiles_m=2, tiles_n=2, stages=2, lds_a=True)
            if fp4_a
            else MfmaTile(tiles_m=2, tiles_n=1, stages=3, lds_a=True)
        )
    elif shape_m <= 64:
        tile = MfmaTile(tiles_m=4, tiles_n=2 if fp4_a else 1, stages=3, lds_a=True)
    elif fp4_a:
        if math.ceil(shape_n / 128) * math.ceil(shape_m / 128) <= num_cus:
            tile = MfmaTile(tiles_m=4, tiles_n=2, stages=3, lds_a=True)
        else:
            tile = MfmaTile(warps_m=2, warps_n=2, tiles_m=4, tiles_n=4, stages=2, lds_a=True)
    else:
        blocks_128x64 = math.ceil(shape_n / 64) * math.ceil(shape_m / 128)
        native_flush = plan.b_mode == B_NATIVE and plan.scale_k > 0
        if blocks_128x64 < num_cus:
            tile = MfmaTile(tiles_m=4, tiles_n=1, stages=3, lds_a=True)
        elif native_flush:
            # Native B with FP32 group / block scales (e.g. FP8 128 x 128 blocks): the per-stage
            # activation-scale registers make 128-row tiles spill; 64 x 128 tiles do not.
            tile = MfmaTile(tiles_m=4, tiles_n=2, stages=2, lds_a=True)
        elif blocks_128x64 < 4 * num_cus:
            tile = MfmaTile(tiles_m=8, tiles_n=1, stages=2, lds_a=True)
        else:
            # Dequantized B (table / INT arithmetic) needs the registers of the third stage.
            tile = MfmaTile(tiles_m=8, tiles_n=2, stages=3 if plan.b_mode == B_NATIVE else 2, lds_a=True)
    row_units = 4 if fp4_a else 8
    threads = tile.warps_m * tile.warps_n * 64
    if tile.block_m * row_units % threads:
        tile = dataclasses.replace(tile, lds_a=False)
    return tile


# M intervals (min exclusive, max inclusive); select_f8_tile is evaluated at each upper bound.
F8_M_INTERVALS = (
    (0, 8),
    (8, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
    (2048, 1 << 30),
)


def get_f8_tuning_configs(config: LayerConfig, num_cus: int) -> list:
    plan = plan_mfma_f8(config)
    assert not isinstance(plan, str), plan
    return [
        [low, high, select_f8_tile(high, config.shape_n, num_cus, plan).to_tuning_config()]
        for low, high in F8_M_INTERVALS
    ]


@dataclasses.dataclass(kw_only=True)
class HummingMfmaF8Kernel(KernelRuntime):
    name: ClassVar[str] = "humming_mfma_f8"
    disable_fast_math: ClassVar[bool] = True

    plan: str  # MfmaF8Plan as JSON (plan_to_json)
    a_dtype: str
    c_dtype: str
    warps_m: int = 1
    warps_n: int = 4
    tiles_m: int = 1
    tiles_n: int = 1
    stages: int = 4
    lds_a: bool = False
    max_splits: int = MAX_SPLITS

    def init_kernel(self) -> None:
        plan = MfmaF8Plan(
            **{key: tuple(v) if isinstance(v, list) else v for key, v in json.loads(self.plan).items()}
        )
        tile = MfmaTile(self.warps_m, self.warps_n, self.tiles_m, self.tiles_n, self.stages, self.lds_a)
        c_dtype = dtypes.DataType.from_str(self.c_dtype)
        self.code = CODE_TEMPLATE.render(
            tile=tile,
            p=plan,
            element_c="Float16" if c_dtype == dtypes.float16 else "BFloat16",
            a_dtype_id=dtypes.DataType.from_str(self.a_dtype).id(),
            c_dtype_id=c_dtype.id(),
            max_splits=self.max_splits,
        )
        self.kernel_expr = "humming_mfma_f8<HummingMfmaF8Traits, HummingMfmaF8Tile>"
        self.prepare()
        from humming import ops

        self.kernel_id, self.kernel_name = ops.register_kernel(self.kernel_filename)


def plan_to_json(plan: MfmaF8Plan) -> str:
    return json.dumps(dataclasses.asdict(plan), sort_keys=True)


def get_f8_kernel_args(config: LayerConfig) -> dict:
    plan = plan_mfma_f8(config)
    assert not isinstance(plan, str), plan
    return {"plan": plan_to_json(plan), "a_dtype": str(config.a_dtype), "c_dtype": str(config.c_dtype)}

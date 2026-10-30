"""ROCm gfx950 MFMA weight-only GEMM (``humming/kernel/humming_mfma.cuh``) behind Humming's public API.

On ROCm ``HummingKernel.prepare_kernels`` delegates here: the layer config is validated, the
gfx950 heuristic (``humming/tune/gfx950.py``) picks one tile per M interval, every tile is
JIT-compiled with ``HIPCCCompiler`` through ``KernelRuntime`` and registered with the C++
launcher, which recognises MFMA kernels and launches them on torch's current stream.

Supported: fp16/bf16 activations x unsigned INT1-8 weights (implicit zero point 2^(bits-1) or
explicit uint8 zero points), per-channel or group (multiple of 32) scales, optional bias, dense
GEMM and MoE GEMM (indexed / grouped_contiguous / grouped_masked, the kernel in
``humming_mfma_moe.py``). Everything else raises ``NotImplementedError``.
"""

import dataclasses
import functools
import json
from typing import ClassVar

import jinja2
import torch

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.config.enum import InputQuantizationMode, WeightScale2Type, WeightScaleType
from humming.device import get_device_index
from humming.jit.runtime import KernelRuntime
from humming.tune.gfx950 import MAX_SPLITS, MfmaTile

MFMA_N_ALIGNMENT = 64
MFMA_K_ALIGNMENT = 128

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/humming_mfma.cuh>

using HummingMfmaTile = MfmaTileConfig<
    {{tile.warps_m}}, {{tile.warps_n}}, {{tile.tiles_m}}, {{tile.tiles_n}}, {{tile.stages}},
    {{tile.lds_a | lower}}, {{tile.scale_b | lower}}>;

// Read by the C++ launcher (csrc/launcher/mfma.h); MFMA_BLOCK_M marks an MFMA kernel.
extern "C" __constant__ uint32_t MFMA_BLOCK_M = HummingMfmaTile::kBlockM;
extern "C" __constant__ uint32_t MFMA_BLOCK_N = HummingMfmaTile::kBlockN;
extern "C" __constant__ uint32_t MFMA_NUM_THREADS = HummingMfmaTile::kNumThreads;
extern "C" __constant__ uint32_t MFMA_A_DTYPE_ID = {{a_dtype_id}};
extern "C" __constant__ uint32_t MFMA_NUM_BITS = {{num_bits}};
extern "C" __constant__ uint32_t MFMA_GROUP_SIZE = {{group_size}};
extern "C" __constant__ uint32_t MFMA_HAS_ZERO_POINT = {{has_zero_point | int}};
extern "C" __constant__ uint32_t MFMA_MAX_SPLITS = {{max_splits}};
""")


def get_mfma_group_size(layer_config: LayerConfig) -> int:
    """Kernel group size: 0 for per-channel scales (a single group spanning K counts as one)."""
    group_size = layer_config.weight_scale_group_size
    if layer_config.weight_scale_type == WeightScaleType.CHANNEL or group_size >= layer_config.shape_k:
        return 0
    return group_size


def get_mfma_unsupported_reason(layer_config: LayerConfig) -> str | None:
    if layer_config.a_dtype.num_bits < 16:
        # Low-bit activations: humming_mfma_f8 (humming/kernel/humming_mfma_f8.py).
        from humming.kernel.humming_mfma_f8 import get_mfma_f8_unsupported_reason

        return get_mfma_f8_unsupported_reason(layer_config)
    if layer_config.a_dtype not in (dtypes.float16, dtypes.bfloat16):
        return f"activation dtype {layer_config.a_dtype} (only float16/bfloat16 activations)"
    if layer_config.c_dtype != layer_config.a_dtype:
        return f"output dtype {layer_config.c_dtype} != activation dtype {layer_config.a_dtype}"
    quant_mode = layer_config.input_quant_mode
    if quant_mode is not None and InputQuantizationMode(quant_mode).should_quantize:
        return f"input quantization ({quant_mode})"
    b_dtype = layer_config.b_dtype
    if not b_dtype.is_integer_type or b_dtype.is_signed or not 1 <= b_dtype.num_bits <= 8:
        return f"weight dtype {b_dtype} (only unsigned/offset-binary integer weights uint1..uint8)"
    if layer_config.weight_scale_type not in (WeightScaleType.CHANNEL, WeightScaleType.GROUP):
        return f"weight scale type {layer_config.weight_scale_type} (only channel/group scales)"
    if layer_config.weight_scale_2_type != WeightScale2Type.NONE:
        return f"secondary weight scales ({layer_config.weight_scale_2_type})"
    if layer_config.bs_dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32):
        return f"weight scale dtype {layer_config.bs_dtype}"
    if layer_config.use_int_weight_scale or layer_config.use_fused_e8m0_scale:
        return "integer / fused E8M0 weight scales"
    if layer_config.has_zero_point and layer_config.is_fp_zero_point:
        return "floating-point zero points"
    group_size = get_mfma_group_size(layer_config)
    if group_size % 32 != 0 or (group_size and layer_config.shape_k % group_size != 0):
        group = layer_config.weight_scale_group_size
        return f"weight scale group size {group} (must be a multiple of 32 dividing K)"
    if layer_config.shape_n % MFMA_N_ALIGNMENT or layer_config.shape_k % MFMA_K_ALIGNMENT:
        return (
            f"padded shape N={layer_config.shape_n}, K={layer_config.shape_k} "
            f"(the ROCm transform pads N to {MFMA_N_ALIGNMENT} and K to {MFMA_K_ALIGNMENT})"
        )
    return None


def check_mfma_layer_config(layer_config: LayerConfig) -> None:
    reason = get_mfma_unsupported_reason(layer_config)
    if reason is not None:
        raise NotImplementedError(f"Humming on ROCm (gfx950 MFMA GEMM) does not support {reason}.")


def check_mfma_gemm_type(layer_config: LayerConfig, gemm_type) -> GemmType:
    """Dense layers run the dense kernel, MoE layers (num_experts > 0) one of the MoE layouts."""
    gemm_type = GemmType.DENSE if gemm_type is None else GemmType(gemm_type)
    if layer_config.num_experts and gemm_type == GemmType.DENSE:
        raise NotImplementedError(
            "Humming on ROCm runs MoE layers (num_experts > 0) as an indexed / grouped_contiguous / "
            "grouped_masked GEMM; pass compute_config={'gemm_type': ...}."
        )
    if not layer_config.num_experts and gemm_type != GemmType.DENSE:
        raise NotImplementedError(f"gemm_type={gemm_type.value} needs an MoE layer (num_experts > 0).")
    return gemm_type


def check_mfma_compute_config(compute_config: dict) -> None:
    for name in ("use_f16_accum", "use_batch_invariant", "use_m_major_input_scale"):
        if compute_config.get(name):
            raise NotImplementedError(f"Humming on ROCm does not support compute_config {name}=True.")


@dataclasses.dataclass(kw_only=True)
class HummingMfmaKernel(KernelRuntime):
    name: ClassVar[str] = "humming_mfma"
    # The kernel was tuned without fast-math flags (tests/rocm/hip_build.py).
    disable_fast_math: ClassVar[bool] = True

    a_dtype: str
    num_bits: int
    group_size: int
    has_zero_point: bool
    warps_m: int = 1
    warps_n: int = 4
    tiles_m: int = 1
    tiles_n: int = 1
    stages: int = 4
    lds_a: bool = False
    scale_b: bool = False
    max_splits: int = MAX_SPLITS

    def init_kernel(self) -> None:
        a_dtype = dtypes.DataType.from_str(self.a_dtype)
        assert a_dtype in (dtypes.float16, dtypes.bfloat16)
        assert 1 <= self.num_bits <= 8 and self.group_size % 32 == 0
        tile = MfmaTile(
            self.warps_m, self.warps_n, self.tiles_m, self.tiles_n, self.stages, self.lds_a, self.scale_b
        )
        self.code = CODE_TEMPLATE.render(
            tile=tile,
            a_dtype_id=a_dtype.id(),
            num_bits=self.num_bits,
            group_size=self.group_size,
            has_zero_point=self.has_zero_point,
            max_splits=self.max_splits,
        )
        element_a = "Float16" if a_dtype == dtypes.float16 else "BFloat16"
        has_zero_point = "true" if self.has_zero_point else "false"
        tile_args = f"HummingMfmaTile, {self.group_size}, {has_zero_point}"
        template_args = f"{element_a}, UInt{self.num_bits}, {tile_args}"
        self.kernel_expr = f"humming_mfma<{template_args}>"
        self.prepare()
        from humming import ops

        self.kernel_id, self.kernel_name = ops.register_kernel(self.kernel_filename)


def _tuning_entries(tuning_obj) -> list:
    """Normalise a tuning config to ``[[min_m, max_m, config], ...]``."""
    if isinstance(tuning_obj, dict):
        tuning_obj = [[0, 1 << 30, tuning_obj]]
    entries = []
    for entry in tuning_obj:
        low, high, config = entry
        if config.get("mma_type") != "mfma":
            raise NotImplementedError(
                "Humming on ROCm only runs MFMA tuning configs "
                "(mma_type='mfma', see humming/tune/gfx950.py); "
                f"got {config}"
            )
        entries.append([low, high, config])
    return entries


def resolve_mfma_configs(layer_config, compute_config, tuning_config, device=None):
    """ROCm counterpart of ``HummingKernel._resolve_configs``."""
    from humming.kernel.humming import HummingKernel
    from humming.tune import get_heuristics_config

    device_index = get_device_index(device)
    layer_obj = dict(HummingKernel._prepare_config_obj(layer_config))
    compute_obj = dict(HummingKernel._prepare_config_obj(compute_config))
    tuning_obj = HummingKernel._prepare_config_obj(tuning_config)
    layer_obj.pop("sublayer_name", None)
    check_mfma_compute_config(compute_obj)
    with torch.cuda.device(device_index):
        layer_config_obj = LayerConfig(**layer_obj)
        layer_config_obj.check_device(device_index)
        check_mfma_layer_config(layer_config_obj)
        gemm_type = check_mfma_gemm_type(layer_config_obj, compute_obj.get("gemm_type"))
        layer_obj["sm_version"] = layer_config_obj.sm_version
        if not tuning_obj:
            tuning_obj = get_heuristics_config(
                layer_config_obj, gemm_type=gemm_type.value, device=device_index
            )
    return layer_obj, compute_obj, _tuning_entries(tuning_obj)


def prepare_mfma_kernels(layer_config, compute_config, tuning_config, device=None) -> torch.Tensor:
    """Compile/register the kernels of every M interval; returns the launcher's int64 config tensor."""
    device_index = get_device_index(device)
    layer_obj, compute_obj, entries = resolve_mfma_configs(
        layer_config, compute_config, tuning_config, device_index
    )
    layer_config_obj = LayerConfig(**layer_obj)
    kernel_cls = HummingMfmaKernel
    gemm_type = check_mfma_gemm_type(layer_config_obj, compute_obj.get("gemm_type"))
    tile_fields = {field.name for field in dataclasses.fields(MfmaTile)} | {"max_splits"}
    if layer_config_obj.a_dtype.num_bits < 16:
        from humming.kernel.humming_mfma_f8 import HummingMfmaF8Kernel, get_f8_kernel_args

        kernel_cls = HummingMfmaF8Kernel
        common = get_f8_kernel_args(layer_config_obj)
        tile_fields = tile_fields - {"scale_b"}
    else:
        common = {
            "a_dtype": str(layer_config_obj.a_dtype),
            "num_bits": layer_config_obj.b_dtype.num_bits,
            "group_size": get_mfma_group_size(layer_config_obj),
            "has_zero_point": layer_config_obj.has_zero_point,
        }
        if gemm_type != GemmType.DENSE:
            from humming.kernel.humming_mfma_moe import HummingMfmaMoeKernel

            kernel_cls = HummingMfmaMoeKernel
            common["gemm_type"] = gemm_type.value
    specs = []
    for _, _, config in entries:
        kernel_args = common | {key: value for key, value in config.items() if key in tile_fields}
        specs.append((kernel_cls, kernel_args))
    kernels = KernelRuntime.compile_many(specs, device_index)
    num_cus = 0  # the launcher queries the CU count
    values = []
    for (low, high, config), kernel in zip(entries, kernels, strict=True):
        values += [low, high, kernel.kernel_id, config.get("num_sms", num_cus)]
    return torch.tensor(values, dtype=torch.int64, device="cpu")


@torch.compiler.assume_constant_result
def get_mfma_gemm_meta(layer_config: str) -> tuple[int, int, int]:
    """``(padded shape_n, pad_shape_n, pad_shape_k)`` of a ROCm layer config string."""
    return _parse_mfma_gemm_meta(layer_config)


@functools.lru_cache(maxsize=1024)
def _parse_mfma_gemm_meta(layer_config: str) -> tuple[int, int, int]:
    layer_obj = json.loads(layer_config)
    return (
        int(layer_obj["shape_n"]),
        int(layer_obj.get("pad_shape_n", 0)),
        int(layer_obj.get("pad_shape_k", 0)),
    )


@torch.compiler.assume_constant_result
def get_mfma_num_experts(layer_config: str) -> int:
    """``num_experts`` of a ROCm layer config string (0 for dense layers)."""
    return _parse_mfma_num_experts(layer_config)


@functools.lru_cache(maxsize=1024)
def _parse_mfma_num_experts(layer_config: str) -> int:
    return int(json.loads(layer_config).get("num_experts") or 0)

import dataclasses
import math
from typing import Literal

import numpy as np

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType, TuningConfig
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.tune.base import DeviceHeuristics
from humming.tune.candidate import (
    CandidateAnalysis,
    ScheduleCandidate,
    TuningDecision,
    TuningProblem,
    analyze_candidate,
    fit_pipeline_stages,
    get_problem_rejection_reasons,
)
from humming.tune.raster import get_moe_raster_group_m
from humming.utils.smem import estimate_smem_size_layer

PACKED_FP4_GROUPED_MIN_SHAPE_M = 768
PACKED_FP4_GROUPED_MAX_ROWS_PER_EXPERT = 8


def fit_sm90_tile(
    layer_config: LayerConfig,
    config: dict,
    use_f16_accum: bool = False,
    *,
    gemm_type: GemmType = GemmType.DENSE,
    use_m_major_input_scale: bool = False,
) -> dict:
    use_wgmma_ss = layer_config.a_dtype.num_bits == layer_config.b_dtype.num_bits
    config = config.copy()
    requested_ctas = config.get("num_ctas_per_sm", 1)
    block_m, block_n, block_k = config["block_shape"]
    warp_m, warp_n, warp_k = config["warp_shape"]
    has_small_group_reduction = layer_config.is_group_input_scale and warp_m <= 16
    if has_small_group_reduction and config.get("use_warp_spec", False) and block_k > warp_k:
        config["block_shape"] = (block_m, block_n, warp_k)
    mma_type = MmaType(config.get("mma_type") or get_default_mma_type(layer_config))
    use_warp_spec = config.get("use_warp_spec", False)
    has_group_weight_scale = layer_config.is_group_weight_scale or layer_config.is_block_weight_scale
    has_weight_scale_accumulator = has_group_weight_scale and not layer_config.use_fused_e8m0_scale
    has_scaled_accumulator = layer_config.is_group_input_scale or has_weight_scale_accumulator
    has_group_accumulator = layer_config.a_dtype.num_bits < 16 and has_scaled_accumulator
    has_packed_input_scales = layer_config.use_packed_k_layout and layer_config.is_group_input_scale
    has_full_group_accumulator = layer_config.a_dtype.is_integer_type
    has_full_group_accumulator |= config.get("wgmma_split_issue_wait", False)
    max_group_warp_m = 80 if has_packed_input_scales and has_full_group_accumulator else 96
    has_unpacked_rs = not use_wgmma_ss and not layer_config.use_packed_k_layout
    should_widen_rs = has_unpacked_rs and has_group_accumulator and not use_warp_spec and warp_n < 32
    if mma_type == MmaType.WGMMA and should_widen_rs and layer_config.shape_n % 128 == 0:
        config["block_shape"] = (block_m, max(128, block_n), block_k)
        config["warp_shape"] = (warp_m, 32, warp_k)

    tuning_fields = TuningConfig.get_field_names()
    while True:
        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        math_threads = math.prod(config["block_shape"]) // math.prod(config["warp_shape"]) * 32
        if mma_type == MmaType.WGMMA:
            if layer_config.use_packed_k_layout and layer_config.is_group_input_scale and math_threads >= 256:
                requested_ctas = min(requested_ctas, 2)
            has_wide_a16_tile = layer_config.a_dtype.num_bits == 16 and warp_n >= 64 and warp_m > 32
            if has_wide_a16_tile and math_threads >= 256:
                requested_ctas = 1
            if "num_ctas_per_sm" in config:
                config["num_ctas_per_sm"] = requested_ctas
            max_warp_m = 192 if warp_n <= 16 else 128 if warp_n <= 32 else 64
            if layer_config.a_dtype.num_bits == 16 and use_warp_spec and warp_n <= 32:
                max_warp_m = 192
            if has_group_accumulator:
                max_warp_m = 160 if warp_n <= 16 else max_group_warp_m if warp_n <= 32 else 48
                if has_unpacked_rs:
                    max_warp_m = 128 if warp_n <= 16 else 80 if warp_n <= 32 else 48
                    if layer_config.a_dtype.is_integer_type:
                        max_warp_m = min(max_warp_m, 64)
                    if layer_config.is_group_input_scale and warp_n >= 32:
                        max_warp_m = min(max_warp_m, 64)
                    if not use_warp_spec:
                        max_warp_m = min(max_warp_m, 8 if warp_n < 32 else 64)
                elif layer_config.use_packed_k_layout and requested_ctas > 1:
                    max_warp_m = min(max_warp_m, 128)
            if math_threads > 256:
                max_warp_m = min(max_warp_m, 32 if has_group_accumulator else 64)
            if requested_ctas >= 3:
                max_warp_m = min(max_warp_m, 32 if use_warp_spec and has_group_accumulator else 64)
            if layer_config.use_packed_k_layout and warp_k > 128:
                max_warp_m = min(max_warp_m, 96)
        else:
            max_warp_m = 64 if warp_n <= 32 else 32
            if has_group_accumulator:
                max_warp_m = min(max_warp_m, 32)
        if warp_m > max_warp_m:
            alignment = 16 if layer_config.a_dtype.is_integer_type and warp_m > 32 else 8
            smaller_m = max_warp_m // alignment * alignment
            config["block_shape"] = (block_m // warp_m * smaller_m, block_n, block_k)
            config["warp_shape"] = (smaller_m, warp_n, warp_k)
            continue

        tuning_values = {key: value for key, value in config.items() if key in tuning_fields}
        for num_ctas in range(requested_ctas, 0, -1):
            tuning = TuningConfig(**(tuning_values | {"num_ctas_per_sm": num_ctas}))
            stages = config.get("num_stages", 2)
            has_packed_group_input = (
                layer_config.use_packed_k_layout
                and layer_config.is_group_input_scale
                and layer_config.shape_k <= 512
            )
            should_limit_stages = has_packed_group_input or (tuning.use_stream_k and use_wgmma_ss)
            mma_type = tuning.mma_type or get_default_mma_type(layer_config)
            if should_limit_stages and mma_type == MmaType.WGMMA:
                k_iters = layer_config.shape_k // config["block_shape"][2]
                stages = min(stages, max(3, k_iters))
            use_tma_c = tuning.use_tma_c and gemm_type != GemmType.INDEXED
            while True:
                smem_size = estimate_smem_size_layer(
                    layer_config,
                    config["block_shape"],
                    gemm_type,
                    stages,
                    warp_shape=config["warp_shape"],
                    smem_reuse_mode=config.get("smem_reuse_mode", "all_stages"),
                    use_tma=config.get("use_tma", False),
                    use_tma_c=use_tma_c,
                    raster_group_m=config.get("raster_group_m", 1),
                    output_chunk_rows=config.get("output_chunk_rows", 0),
                    use_warp_spec=config.get("use_warp_spec", False),
                    mma_type=tuning.mma_type or get_default_mma_type(layer_config),
                    mma_accum_bits=16 if use_f16_accum else 32,
                    use_m_major_input_scale=use_m_major_input_scale,
                )
                if smem_size * num_ctas <= 227 * 1024 or stages <= 3:
                    break
                stages -= 1
            if smem_size * num_ctas > 227 * 1024:
                continue
            if num_ctas != requested_ctas:
                config["num_ctas_per_sm"] = num_ctas
            if stages != config.get("num_stages", 2):
                config["num_stages"] = stages
            return config

        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        math_threads = math.prod(config["block_shape"]) // math.prod(config["warp_shape"]) * 32
        if math_threads > 256 and block_k > warp_k:
            config["block_shape"] = (block_m, block_n, block_k // 2)
            continue
        alignment = 16 if layer_config.a_dtype.is_integer_type and warp_m > 32 else 8
        smaller_m = min(warp_m - 1, max_warp_m) // alignment * alignment
        if smaller_m < 8:
            raise ValueError(f"no SM90 tile fits shared memory: {config}")
        config["block_shape"] = (block_m // warp_m * smaller_m, block_n, block_k)
        config["warp_shape"] = (smaller_m, warp_n, warp_k)


def select_sm90_small_m_schedule(
    problem: TuningProblem,
    selected: TuningDecision | dict,
    *,
    is_h20: bool = False,
) -> TuningDecision | dict:
    layer = problem.layer_config
    config = selected.to_config() if isinstance(selected, TuningDecision) else selected
    num_sms = problem.device.num_sms
    assert num_sms is not None
    block_m, block_n, block_k = config["block_shape"]
    weight_bits = layer.b_dtype.num_bits
    is_power_of_two_weight = weight_bits & (weight_bits - 1) == 0
    warp_n, warp_k = (64, 32) if is_power_of_two_weight else (32, 64)
    mma_m = 8 if block_m <= 8 else math.ceil(block_m / 16) * 16
    mma_k = 128
    num_stages = 4
    num_weight_tiles = (layer.shape_n // 128) * (layer.shape_k // 128)
    reason = None
    proposal = None

    if problem.gemm_type == GemmType.DENSE:
        has_two_cta_work = num_weight_tiles >= num_sms * 2 * num_stages * 2
        has_small_accumulator = mma_m * warp_n <= 1024
        num_ctas = 2 if has_two_cta_work and has_small_accumulator else 1
        has_pipeline_work = num_weight_tiles >= num_sms * num_stages
        has_small_m = problem.shape_m <= 16
        has_decode_work = problem.shape_m <= 32 and weight_bits >= 4 and has_two_cta_work
        has_decode_work &= mma_m == block_m
        if has_pipeline_work and (has_small_m or has_decode_work):
            reason = "selected MMA with CTA residency fitted to K work and accumulator size"
    elif mma_m == 8:
        num_output_tiles = problem.estimate_num_blocks_m(mma_m) * (layer.shape_n // 128)
        if is_h20 and num_output_tiles > num_sms:
            return selected
        mma_k = 64
        num_ctas = 4
        reason = "selected MMA for eight-row expert tiles"
    elif mma_m <= 32:
        # Halving an indexed K tile adds loads without removing producer overhead.
        if problem.gemm_type == GemmType.INDEXED and block_k > 128:
            return selected
        num_output_tiles = problem.estimate_num_blocks_m(mma_m) * (layer.shape_n // 128)
        num_base_tiles = problem.estimate_num_blocks_m(block_m) * (layer.shape_n // block_n)
        has_short_grid = num_output_tiles <= num_sms * 5 / 4
        has_extra_k_parallelism = block_k <= 64 and num_output_tiles <= num_sms * 9 / 4
        has_no_extra_padding = mma_m == block_m and num_output_tiles <= num_sms * 2
        can_round_m = block_m <= 16 and (has_short_grid or has_extra_k_parallelism or has_no_extra_padding)
        # Rounding 24-row expert tiles to 32 regresses with skewed routing.
        if can_round_m:
            num_ctas = 2 if mma_m == 16 and num_output_tiles >= num_sms / 2 else 1
            reason = "selected MMA for a short expert grid with bounded row padding"
        elif block_m < 32 and block_k >= 128 and num_base_tiles <= num_sms:
            if config.get("use_tma", False) or config.get("use_warp_spec", False):
                proposal = config | {"use_tma": False, "use_warp_spec": False}
                reason = "removed producer and TMA overhead from a short expert grid"

    if reason is None:
        return selected
    if proposal is None:
        proposal = {
            "mma_type": MmaType.MMA.value,
            "block_shape": (mma_m, 128, mma_k),
            "warp_shape": (mma_m, warp_n, warp_k),
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas,
            "use_stream_k": True,
            "use_tma": False,
            "use_warp_spec": False,
        }
    candidate = ScheduleCandidate.from_config("sm90_small_m", proposal)
    analysis = analyze_candidate(problem, candidate)
    if not analysis.legal:
        return selected
    if isinstance(selected, TuningDecision):
        considered = selected.considered
    else:
        base_config = {key: value for key, value in config.items() if key != "num_sms"}
        base = ScheduleCandidate.from_config("sm90_small_m_base", base_config)
        considered = (analyze_candidate(problem, base),)
    return TuningDecision(
        problem=problem,
        family="sm90_small_m",
        selected=candidate,
        considered=(*considered, analysis),
        reason=reason,
    )


def get_sm90_specialized_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    def finalize_config(config: dict | None) -> dict | None:
        if config is None or gemm_type == GemmType.DENSE or not config.get("use_stream_k", False):
            return config
        block_m, block_n, _ = config["block_shape"]
        counts = DeviceHeuristics.get_expert_rows(layer_config, shape_m, gemm_type, expert_probability_cv)
        num_m_tiles = ((counts + block_m - 1) // block_m).sum(axis=1).mean()
        num_output_tiles = num_m_tiles * math.ceil(layer_config.shape_n / block_n)
        resident_ctas = current_device.sm_count * config.get("num_ctas_per_sm", 1)
        # Large grids amortize the tail imbalance without splitting the K reduction.
        if num_output_tiles >= 16 * resident_ctas:
            return config | {"use_stream_k": False}
        return config

    compact_config = get_compact_scaled_moe_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        use_m_major_input_scale=use_m_major_input_scale,
        expert_probability_cv=expert_probability_cv,
    )
    if compact_config is not None:
        return finalize_config(compact_config)
    short_config = get_short_k_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        use_m_major_input_scale=use_m_major_input_scale,
    )
    if short_config is not None:
        return finalize_config(short_config)
    block_scaled_config = get_block_scaled_moe_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        expert_probability_cv=expert_probability_cv,
    )
    if block_scaled_config is not None:
        return finalize_config(block_scaled_config)
    scaled_config = get_scaled_a8_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        use_m_major_input_scale=use_m_major_input_scale,
        expert_probability_cv=expert_probability_cv,
    )
    is_grouped = gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED)
    is_moe = is_grouped or gemm_type == GemmType.INDEXED
    if is_moe and scaled_config is not None:
        if scaled_config.get("num_ctas_per_sm", 1) > 1 and not scaled_config["use_warp_spec"]:
            return finalize_config(scaled_config)
    overlap_config = get_overlap_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        use_m_major_input_scale=use_m_major_input_scale,
        expert_probability_cv=expert_probability_cv,
    )
    if overlap_config is not None:
        return finalize_config(overlap_config)
    packed_config = get_packed_wna8_config(
        layer_config,
        shape_m,
        gemm_type,
        use_f16_accum,
        use_batch_invariant,
        is_h20=is_h20,
        use_m_major_input_scale=use_m_major_input_scale,
        expert_probability_cv=expert_probability_cv,
    )
    if packed_config is not None:
        return finalize_config(packed_config)
    return finalize_config(scaled_config)


def get_small_m_packed_fp4_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
) -> dict | None:
    """Use non-Stream-K tiles for bandwidth-bound grouped FP4 experts.

    At one to eight routed rows per expert, the generic grouped-scale policy
    spends one or more waves on small N tiles and pays Stream-K fixups. A
    wider N tile amortizes packed-FP4 dequantization. BF16 activations use
    MMA to stay within the register budget; FP8 activations use a WGMMA tile
    with TMA and warp specialization.
    """
    has_supported_runtime = (
        not is_h20
        and not use_f16_accum
        and not use_batch_invariant
        and not use_m_major_input_scale
        and gemm_type == GemmType.GROUPED_CONTIGUOUS
    )
    has_supported_experts = bool(layer_config.num_experts)
    has_supported_weight = (
        layer_config.b_dtype == dtypes.float4e2m1 and layer_config.weight_scale_group_size == 32
    )
    has_supported_layout = (
        not layer_config.use_packed_k_layout and not layer_config.has_zero_point and not layer_config.has_bias
    )
    is_tile_aligned = layer_config.shape_n % 256 == 0 and layer_config.shape_k % 128 == 0
    # vLLM reports the padded CUDA-graph bucket as shape_m. Below this bound, a
    # small real batch can leave the routed layout extremely sparse.
    has_supported_shape_m = (
        PACKED_FP4_GROUPED_MIN_SHAPE_M
        <= shape_m
        <= PACKED_FP4_GROUPED_MAX_ROWS_PER_EXPERT * layer_config.num_experts
    )
    if not (
        has_supported_runtime
        and has_supported_experts
        and has_supported_weight
        and has_supported_layout
        and is_tile_aligned
        and has_supported_shape_m
    ):
        return None

    if layer_config.a_dtype == dtypes.bfloat16 and layer_config.input_scale_group_size == 0:
        if layer_config.shape_n >= 2 * layer_config.shape_k:
            return {
                "mma_type": MmaType.MMA.value,
                "block_shape": (8, 256, 128),
                "warp_shape": (8, 64, 64),
                "num_stages": 4,
                "num_ctas_per_sm": 2,
                "use_stream_k": False,
                "use_tma": False,
                "use_warp_spec": False,
            }
        return {
            "mma_type": MmaType.MMA.value,
            "block_shape": (8, 128, 128),
            "warp_shape": (8, 64, 64),
            "num_stages": 4,
            "num_ctas_per_sm": 4,
            "use_stream_k": False,
            "use_tma": False,
            "use_warp_spec": False,
        }

    if (
        layer_config.a_dtype == dtypes.float8e4m3
        and layer_config.input_scale_group_size == 32
        and layer_config.use_fused_e8m0_scale
    ):
        return {
            "mma_type": MmaType.WGMMA.value,
            "block_shape": (8, 256, 128),
            "warp_shape": (8, 32, 128),
            "num_stages": 4,
            "num_ctas_per_sm": 2,
            "use_stream_k": False,
            "use_tma": True,
            "use_warp_spec": True,
        }

    return None


def get_compact_scaled_moe_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    use_wgmma_ss = layer_config.a_dtype.num_bits == layer_config.b_dtype.num_bits
    if gemm_type == GemmType.DENSE or use_f16_accum or use_batch_invariant:
        return None
    if layer_config.a_dtype.num_bits != 8 or layer_config.use_packed_k_layout:
        return None
    if use_wgmma_ss or layer_config.shape_n % 64 or layer_config.shape_k % 64:
        return None
    has_short_k = layer_config.shape_k <= 512
    has_narrow_n = layer_config.shape_n % 128 != 0
    if not (has_short_k or has_narrow_n):
        return None
    if has_narrow_n and layer_config.shape_n // 64 >= current_device.sm_count / 4:
        return None
    if get_problem_rejection_reasons(layer_config, MmaType.WGMMA):
        return None

    has_group_accumulator = layer_config.is_group_input_scale or (
        not layer_config.use_fused_e8m0_scale and layer_config.is_group_weight_scale
    )
    if not has_group_accumulator or (is_h20 and not has_narrow_n):
        return None
    if not has_narrow_n and layer_config.input_scale_group_size >= 128:
        return None
    has_grouped_token_scale = gemm_type != GemmType.INDEXED and not layer_config.is_group_input_scale
    if not is_h20 and has_grouped_token_scale and (has_narrow_n or layer_config.shape_k >= 512):
        return None
    nominal_block_m = select_sm90_block_m(layer_config, shape_m, 160, gemm_type, expert_probability_cv)
    if not has_narrow_n and gemm_type == GemmType.INDEXED and nominal_block_m <= 32:
        return None
    if not has_narrow_n and layer_config.shape_k >= 512 and nominal_block_m <= 80:
        return None
    has_indexed_token_scale = gemm_type == GemmType.INDEXED and not layer_config.is_group_input_scale
    min_narrow_m = 16 if is_h20 and has_indexed_token_scale else 32
    if has_narrow_n and nominal_block_m < min_narrow_m:
        return None
    max_block_m = 64 if layer_config.a_dtype.is_integer_type else 80
    if is_h20 or layer_config.is_group_input_scale or has_narrow_n:
        max_block_m = 32 if has_narrow_n else 64
    block_m = select_sm90_block_m(layer_config, shape_m, max_block_m, gemm_type, expert_probability_cv)
    block_n = 64 if has_narrow_n else 128
    block_k = 64
    use_warp_spec = block_m <= 64 and (block_m > 32 or has_narrow_n)
    config = {
        "mma_type": MmaType.WGMMA.value,
        "block_shape": (block_m, block_n, block_k),
        "warp_shape": (block_m, block_n // 4, block_k),
        "num_stages": 3,
        "num_ctas_per_sm": 3 if is_h20 and has_narrow_n else 2,
        "use_tma": use_warp_spec,
        "use_warp_spec": use_warp_spec,
        "use_stream_k": True,
        "smem_reuse_mode": "none" if use_warp_spec else "all_stages",
        "producer_stage_unroll": 2 if use_warp_spec else 3,
        "wgmma_split_issue_wait": use_warp_spec,
    }
    return fit_sm90_tile(
        layer_config,
        config,
        gemm_type=gemm_type,
        use_m_major_input_scale=use_m_major_input_scale,
    )


def get_compact_moe_config(
    layer_config: LayerConfig,
    config: dict,
    gemm_type: GemmType,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    *,
    use_m_major_input_scale: bool = False,
) -> dict:
    is_wna16 = layer_config.a_dtype.num_bits == 16 and layer_config.b_dtype.num_bits < 16
    if not is_wna16 or gemm_type == GemmType.DENSE or use_f16_accum or use_batch_invariant:
        return config
    mma_type = MmaType(config.get("mma_type") or get_default_mma_type(layer_config))
    if not config.get("use_warp_spec", False) or mma_type != MmaType.WGMMA:
        return config

    block_m, block_n, block_k = config["block_shape"]
    has_single_warpgroup = tuple(config["warp_shape"]) == (block_m, block_n // 4, block_k)
    has_two_warpgroups = tuple(config["warp_shape"]) == (block_m, block_n // 8, block_k)
    is_grouped = gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED)
    can_compact_single = has_single_warpgroup and block_m <= 32
    has_grouped_warpgroups = is_grouped and (has_single_warpgroup or has_two_warpgroups)
    can_compact_grouped = has_grouped_warpgroups and not can_compact_single
    if block_k != 64 or not (can_compact_single or can_compact_grouped):
        return config

    num_ctas = max(2, config.get("num_ctas_per_sm", 1))
    candidate = config | {
        "num_ctas_per_sm": num_ctas,
        "use_warp_spec": False,
        "use_tma": False,
        "use_tma_c": False,
    }
    if can_compact_grouped:
        candidate["num_stages"] = max(6, config["num_stages"])
    else:
        candidate |= {
            "smem_reuse_mode": "all_stages",
            "producer_stage_unroll": config["num_stages"],
            "wgmma_split_issue_wait": False,
        }
    candidate = fit_sm90_tile(
        layer_config,
        candidate,
        gemm_type=gemm_type,
        use_m_major_input_scale=use_m_major_input_scale,
    )
    keeps_tile = tuple(candidate["block_shape"]) == tuple(config["block_shape"])
    keeps_warp = tuple(candidate["warp_shape"]) == tuple(config["warp_shape"])
    keeps_residency = candidate["num_ctas_per_sm"] == num_ctas
    keeps_pipeline = not can_compact_grouped or candidate["num_stages"] >= config["num_stages"]
    return candidate if keeps_tile and keeps_warp and keeps_residency and keeps_pipeline else config


def get_overlap_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    layer = layer_config
    use_wgmma_ss = layer.a_dtype.num_bits == layer.b_dtype.num_bits
    is_dense = gemm_type == GemmType.DENSE
    is_masked = gemm_type == GemmType.GROUPED_MASKED
    is_a16 = layer.a_dtype.num_bits == 16
    is_unpacked_a8 = layer.a_dtype.num_bits == 8 and not use_wgmma_ss and not layer.use_packed_k_layout
    can_overlap_a16 = is_a16 and not is_dense and (not is_h20 or is_masked) and layer.b_dtype.num_bits < 16
    can_overlap_packed = layer.use_packed_k_layout and not is_h20 and (not is_dense or shape_m <= 64)
    can_overlap_block = use_wgmma_ss and layer.is_block_weight_scale and not is_dense and not is_h20
    can_overlap_small_ss = use_wgmma_ss and is_dense and not is_h20 and shape_m <= 64
    has_a8_schedule = is_unpacked_a8 or can_overlap_packed or can_overlap_block or can_overlap_small_ss
    if not (can_overlap_a16 or (layer.a_dtype.num_bits == 8 and has_a8_schedule)):
        return None
    if use_f16_accum or use_batch_invariant or layer.shape_k <= 512:
        return None
    block_k = 64 if is_a16 else 128
    if layer.shape_n % 128 or layer.shape_k % block_k:
        return None
    if get_problem_rejection_reasons(layer, MmaType.WGMMA):
        return None
    if is_dense and shape_m <= (64 if is_h20 else 32):
        return None
    if is_dense and shape_m <= 64 and layer.shape_k > 1024:
        return None

    has_separate_group_scales = layer.is_group_input_scale and not layer.use_fused_e8m0_scale
    has_long_narrow_reduction = layer.shape_k > 4096 and layer.shape_n <= 1024
    if not is_h20 and has_separate_group_scales and has_long_narrow_reduction:
        return None

    num_sms = current_device.sm_count
    if is_dense:
        distributions = [np.asarray([[shape_m]])]
    else:
        distributions = [
            DeviceHeuristics.get_expert_rows(layer, shape_m, gemm_type, cv)
            for cv in (0.0, expert_probability_cv, max(1.0, expert_probability_cv))
        ]
        full_tiles = (distributions[1] // 16).sum(axis=1) * (layer.shape_n // 128)
        if full_tiles.mean() < num_sms:
            return None

    can_widen_unpacked = is_unpacked_a8 and not is_dense and not layer.is_group_input_scale
    can_widen_grouped_a16 = can_overlap_a16 and not is_h20 and gemm_type != GemmType.INDEXED
    configs = []
    costs = []
    for block_n in (128, 256):
        if layer.shape_n % block_n:
            continue
        if has_separate_group_scales and has_long_narrow_reduction and block_n == 256:
            continue
        warp_n = max(32 if is_a16 or (is_dense and shape_m <= 64) else 16, block_n // 8)
        if is_dense:
            max_m = 96 if is_h20 else 176
            if layer.is_group_input_scale:
                max_m = min(max_m, 96 if layer.input_scale_group_size < block_k else 128)
            if block_n == 256:
                max_m = min(max_m, 64 if layer.is_group_input_scale else 96)
        else:
            max_m = 64 if is_h20 or layer.is_group_input_scale else 96
            if can_widen_grouped_a16 and shape_m > layer.num_experts * max_m:
                max_m = 176
            if can_overlap_packed:
                max_m = 160 if block_n == 128 else 48
            if can_widen_unpacked:
                max_m = 176 if block_n == 128 else 128
                if gemm_type == GemmType.INDEXED:
                    max_m = min(max_m, 160)
            if is_h20 and gemm_type == GemmType.INDEXED and layer.a_dtype.is_integer_type:
                max_m = 32
        for block_m in range(16, max_m + 1, 8):
            if layer.a_dtype.is_integer_type and block_m > 32 and block_m % 16:
                continue
            use_indexed_overlap = gemm_type == GemmType.INDEXED and block_m <= 32
            use_indexed_overlap &= is_unpacked_a8 or can_overlap_packed
            config = {
                "mma_type": MmaType.WGMMA.value,
                "block_shape": (block_m, block_n, block_k),
                "warp_shape": (block_m, warp_n, block_k),
                "num_stages": 4,
                "num_ctas_per_sm": 2 if use_indexed_overlap else 1,
                "use_warp_spec": True,
                "use_tma": True,
                "use_stream_k": False,
                "smem_reuse_mode": "none",
                "producer_stage_unroll": 1,
                "wgmma_split_issue_wait": True,
                "wgmma_use_late_as": layer.is_group_input_scale and (block_m > 96 or use_indexed_overlap),
                "raster_group_m": 1,
            }
            fitted = fit_sm90_tile(
                layer,
                config,
                gemm_type=gemm_type,
                use_m_major_input_scale=use_m_major_input_scale,
            )
            if fitted != config:
                continue
            smem_size = estimate_smem_size_layer(
                layer,
                config["block_shape"],
                gemm_type,
                4,
                warp_shape=config["warp_shape"],
                smem_reuse_mode="none",
                use_tma=True,
                use_warp_spec=True,
                mma_type=MmaType.WGMMA,
                use_m_major_input_scale=use_m_major_input_scale,
            )
            if smem_size * config["num_ctas_per_sm"] > 227 * 1024:
                continue
            tiles = [
                ((rows + block_m - 1) // block_m).sum(axis=1) * (layer.shape_n // block_n)
                for rows in distributions
            ]
            has_short_grid = is_dense and tiles[0].mean() < num_sms * 2 / 3
            config["use_stream_k"] = bool(not is_dense or (has_short_grid and layer.shape_k >= 2048))
            wide_efficiency = 1.1 if block_n == 256 else 1.0
            if gemm_type == GemmType.INDEXED and can_widen_unpacked and block_n == 256:
                wide_efficiency = 1.15
            has_grouped_weight_conversion = can_widen_unpacked and gemm_type != GemmType.INDEXED
            has_weight_conversion_cost = layer.shape_k > 4096 or has_grouped_weight_conversion
            weight_cost = 64 if not is_h20 and has_weight_conversion_cost else 32
            tile_cost = (block_m + weight_cost) * block_n / 128 / wide_efficiency
            resident_ctas = config["num_ctas_per_sm"]
            waves = [np.ceil(count / (num_sms * resident_ctas)).mean() * resident_ctas for count in tiles]
            if is_dense and config["use_stream_k"]:
                waves = [max(0.65, count.mean() / num_sms) for count in tiles]
                tile_cost *= 1.15
            configs.append(config)
            costs.append([wave * tile_cost for wave in waves])
    if not configs:
        return None
    work = np.asarray(costs)
    relative_work = work / work.min(axis=0)
    scores = relative_work.mean(axis=1)
    config = configs[int(np.argmin(scores))]
    block_m, block_n, _ = config["block_shape"]
    use_a16_raster = can_overlap_a16 and gemm_type != GemmType.INDEXED
    use_wide_raster = gemm_type == GemmType.GROUPED_CONTIGUOUS and block_m >= 48 and block_n == 256
    if use_a16_raster or use_wide_raster:
        config["raster_group_m"] = 4
    is_fp8 = layer.a_dtype in (dtypes.float8e4m3, dtypes.float8e5m2)
    if can_overlap_block and is_fp8 and not is_dense:
        config["raster_group_m"] = get_moe_raster_group_m(shape_m, layer.num_experts, block_m)
    can_widen_indexed_a16 = is_a16 and gemm_type == GemmType.INDEXED and block_n < 512
    if can_widen_indexed_a16 and layer.shape_n % 512 == 0:
        wide_config = config | {
            "block_shape": (block_m, 512, block_k),
            "warp_shape": (block_m, 32, block_k),
        }
        wide_m_tiles = ((distributions[1] + block_m - 1) // block_m).sum(axis=1).mean()
        wide_output_tiles = wide_m_tiles * (layer.shape_n // 512)
        fitted = fit_sm90_tile(layer, wide_config, gemm_type=gemm_type)
        if wide_output_tiles >= num_sms and fitted == wide_config:
            config = wide_config
    return config


def get_short_k_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
) -> dict | None:
    use_wgmma_ss = layer_config.a_dtype.num_bits == layer_config.b_dtype.num_bits
    if gemm_type != GemmType.DENSE or use_f16_accum or use_batch_invariant:
        return None
    if layer_config.a_dtype.num_bits != 8 or layer_config.shape_k > 512:
        return None
    if layer_config.shape_n % 128 or layer_config.shape_k % 128:
        return None
    if get_problem_rejection_reasons(layer_config, MmaType.WGMMA):
        return None
    if is_h20 and use_wgmma_ss and layer_config.shape_k > 256 and shape_m > 64:
        return None
    if layer_config.use_fused_e8m0_scale and layer_config.use_packed_k_layout and shape_m > 64:
        return None

    alignment = 16 if layer_config.a_dtype.is_integer_type else 8
    n_tiles = layer_config.shape_n // 128
    block_m = int(shape_m * n_tiles / current_device.sm_count) // alignment * alignment
    block_m = min(64, max(16, block_m))
    has_unpacked_group_weights = (
        not use_wgmma_ss
        and not layer_config.use_packed_k_layout
        and layer_config.is_group_weight_scale
        and not layer_config.use_fused_e8m0_scale
    )
    warp_n = 32 if has_unpacked_group_weights else 16
    config = {
        "mma_type": MmaType.WGMMA.value,
        "block_shape": (block_m, 128, 128),
        "warp_shape": (block_m, warp_n, 128),
        "num_stages": 3,
        "num_ctas_per_sm": 2,
        "use_tma": False,
        "use_warp_spec": False,
        "use_stream_k": False,
        "smem_reuse_mode": "all_stages",
        "raster_group_m": 1,
    }
    config = fit_sm90_tile(
        layer_config,
        config,
        gemm_type=gemm_type,
        use_m_major_input_scale=use_m_major_input_scale,
    )
    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        gemm_type,
        config["num_stages"],
        warp_shape=config["warp_shape"],
        use_tma=False,
        use_warp_spec=False,
        mma_type=MmaType.WGMMA,
        use_m_major_input_scale=use_m_major_input_scale,
    )
    return config if smem_size * config["num_ctas_per_sm"] <= 227 * 1024 else None


def get_block_scaled_moe_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    fp8_dtypes = (dtypes.float8e4m3, dtypes.float8e5m2)
    has_fp8_operands = layer_config.a_dtype in fp8_dtypes and layer_config.b_dtype == layer_config.a_dtype
    has_block_scales = layer_config.is_block_weight_scale
    is_grouped = gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED)
    has_short_indexed_reduction = is_h20 and gemm_type == GemmType.INDEXED and layer_config.shape_k <= 1024
    if not (has_fp8_operands and has_block_scales and (is_grouped or has_short_indexed_reduction)):
        return None
    if use_f16_accum or use_batch_invariant or not layer_config.num_experts:
        return None
    if get_problem_rejection_reasons(layer_config, MmaType.WGMMA):
        return None

    shape_n, shape_k = layer_config.shape_n, layer_config.shape_k
    if shape_n % 128 or shape_k % 128:
        return None

    # Separate output storage lets the stage ring continue across expert tiles.
    # H20 favors two resident CTAs; the full Hopper compute throughput benefits
    # from a wider tile with more accumulator registers per CTA.
    block_m, block_n, warp_n = (48, 128, 16) if is_h20 else (80, 256, 32)
    stages, resident_ctas = (3, 2) if is_h20 else (4, 1)
    use_warp_spec = True
    use_stream_k = True
    short_k = shape_k <= 512
    has_long_narrow_reduction = is_h20 and shape_n <= 1024 and shape_k >= 4096
    if short_k:
        block_m = 64 if is_h20 else 96
        stages = 3
    elif has_long_narrow_reduction:
        resident_config = fit_sm90_tile(
            layer_config,
            {
                "block_shape": (64, 128, 128),
                "warp_shape": (64, 16, 128),
                "num_stages": stages,
                "num_ctas_per_sm": 3,
                "use_tma": True,
                "use_warp_spec": True,
                "smem_reuse_mode": "none",
            },
            gemm_type=gemm_type,
        )
        block_m, block_n, _ = resident_config["block_shape"]
        warp_n = resident_config["warp_shape"][1]
        stages, resident_ctas = resident_config["num_stages"], resident_config["num_ctas_per_sm"]
    elif is_h20 and shape_k <= 2048:
        stages = 4

    # Narrow output widths cannot form the wider tile. Removing the producer
    # warpgroup makes two smaller CTAs fit without sacrificing register space.
    if shape_n % block_n:
        block_m, block_n, warp_n = 80, 128, 16
        stages, resident_ctas = 3, 2
        use_warp_spec = False

    # Account for per-expert padding and incomplete waves before using larger tiles.
    counts = DeviceHeuristics.get_expert_rows(layer_config, shape_m, gemm_type, expert_probability_cv)
    resident_blocks = current_device.sm_count * resident_ctas
    if short_k:
        block_m_candidates = (80, 96) if block_m == 96 else (block_m,)
    else:
        # Preserve the pipeline crossover while comparing its M tiles.
        base_m_tiles = ((counts + block_m - 1) // block_m).sum(axis=1)
        base_utilization = counts.sum(axis=1) / (base_m_tiles * block_m)
        if base_utilization.mean() < 2 / 3:
            return None
        base_output_tiles = base_m_tiles * (shape_n // block_n)
        has_large_grid = base_output_tiles.mean() >= 16 * resident_blocks
        block_m_candidates = tuple(range(8, block_m + 1, 8)) if has_large_grid else (block_m,)
    best = None
    for candidate_m in block_m_candidates:
        m_tiles = ((counts + candidate_m - 1) // candidate_m).sum(axis=1)
        tiles = m_tiles * (shape_n // block_n)
        utilization = counts.sum(axis=1) / (m_tiles * candidate_m)
        if utilization.mean() < 2 / 3 or tiles.mean() < resident_blocks * 2 / 3:
            continue
        waves = (tiles + resident_blocks - 1) // resident_blocks
        # As in SM100, balance tile rounds against padded math per tile.
        score = float(waves.mean() * math.sqrt(candidate_m))
        if best is None or score < best[0]:
            best = score, candidate_m
    if best is None:
        return None
    block_m = best[1]

    raster_group_m = get_moe_raster_group_m(shape_m, layer_config.num_experts, block_m)

    config = {
        "mma_type": MmaType.WGMMA.value,
        "block_shape": (block_m, block_n, 128),
        "warp_shape": (block_m, warp_n, 128),
        "num_stages": stages,
        "num_ctas_per_sm": resident_ctas,
        "use_stream_k": use_stream_k,
        "use_warp_spec": use_warp_spec,
        "use_tma": True,
        "use_tma_c": use_warp_spec and not short_k,
        "smem_reuse_mode": "none",
        "wgmma_split_issue_wait": not use_warp_spec,
        "raster_group_m": raster_group_m,
    }
    if gemm_type == GemmType.INDEXED:
        config.update(use_tma_a=False, use_tma_c=False)
    if short_k or has_long_narrow_reduction:
        config["producer_stage_unroll"] = 2
        config["consumer_stage_unroll"] = 2 if short_k else 1

    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        gemm_type,
        stages,
        warp_shape=config["warp_shape"],
        smem_reuse_mode="none",
        use_tma=True,
        use_warp_spec=use_warp_spec,
        raster_group_m=raster_group_m,
    )
    if smem_size * resident_ctas > 227 * 1024:
        return None
    return config


@dataclasses.dataclass(frozen=True, slots=True)
class Sm90CandidatePolicy:
    max_indexed_threads_for_two_ctas: int = 256
    max_indexed_threads_for_three_ctas: int = 256


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class _MoeA16Option:
    candidate: ScheduleCandidate
    transform: Literal["base", "half_k", "split_n_widen_k", "mma"]
    priority: int
    parent: "_MoeA16Option | None" = None


def calc_sm90_num_block_list(
    layer_config: LayerConfig,
    shape_m: int,
    max_block_m: int,
    gemm_type: GemmType | None = None,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> list[float]:
    num_blocks_list: list[float] = []
    if gemm_type == GemmType.DENSE or not layer_config.num_experts:
        for block_m in range(8, max_block_m + 1, 8):
            num_blocks_list.append(math.ceil(shape_m / block_m))
    else:
        counts = DeviceHeuristics.get_expert_rows(
            layer_config, shape_m, gemm_type or GemmType.GROUPED_CONTIGUOUS, expert_probability_cv
        )
        for block_m in range(8, max_block_m + 1, 8):
            num_blocks = float(((counts + block_m - 1) // block_m).sum(axis=1).mean())
            num_blocks_list.append(num_blocks)

    for index, block_m in enumerate(range(8, max_block_m + 1, 8)):
        if layer_config.a_dtype == dtypes.int8 and block_m % 16 == 8 and block_m > 32:
            num_blocks_list[index] = 1000000

    return num_blocks_list


def select_sm90_block_m(
    layer_config: LayerConfig,
    shape_m: int,
    max_block_m: int,
    gemm_type: GemmType,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> int:
    num_blocks_list = calc_sm90_num_block_list(
        layer_config,
        shape_m,
        max_block_m,
        gemm_type,
        expert_probability_cv,
    )
    if gemm_type != GemmType.DENSE and layer_config.num_experts:
        num_blocks_list = [
            tiles * (block_m + 32)
            for block_m, tiles in zip(range(8, max_block_m + 1, 8), num_blocks_list, strict=True)
        ]
    return np.argmin(num_blocks_list).item() * 8 + 8


def build_sm90_seed_config(problem: TuningProblem) -> dict:
    """Build the sparse seed config shared by legacy and MoE A16 paths."""
    layer_config = problem.layer_config
    tune_moe_a16 = (
        problem.gemm_type != GemmType.DENSE
        and layer_config.a_dtype.num_bits == 16
        and not problem.use_batch_invariant
    )
    if layer_config.use_packed_k_layout:
        max_block_m = 128
    elif problem.use_f16_accum:
        max_block_m = 256
    else:
        max_block_m = 176

    if tune_moe_a16:
        max_block_m = 64
    block_shape_m = select_sm90_block_m(
        layer_config, problem.shape_m, max_block_m, problem.gemm_type, problem.expert_probability_cv
    )
    warp_shape_n = 32
    warp_shape_k = 1024 // layer_config.a_dtype.num_bits

    # Long-K layers need more routed rows before wider N tiles pay off.
    wide_tile_min_shape_m = 64 if layer_config.shape_k > 4096 else 16
    use_wide_moe_tile = tune_moe_a16 and block_shape_m <= 64 and problem.shape_m >= wide_tile_min_shape_m
    if use_wide_moe_tile:
        warp_shape_n = 64
        # N=512 spills its accumulator at two-CTA residency from M=48 onward.
        if layer_config.shape_k <= 512 and layer_config.shape_n >= 2048 and block_shape_m < 48:
            block_shape_n = 512
            block_shape_k = 64
        else:
            block_shape_n = 256
            block_shape_k = 128
    elif layer_config.shape_n <= 4096 and not problem.use_batch_invariant and block_shape_m <= 64:
        block_shape_n = 128
        block_shape_k = warp_shape_k * 2
        if block_shape_m <= 32:
            block_shape_k = block_shape_k * 2
        if block_shape_k > 256:
            block_shape_k = block_shape_k // 2
            warp_shape_k = warp_shape_k // 2

        while layer_config.shape_k % block_shape_k != 0:
            block_shape_k = block_shape_k // 2
    else:
        block_shape_n = 256
        block_shape_k = warp_shape_k
        if block_shape_m <= 32 and layer_config.b_dtype.num_bits <= 6:
            block_shape_k = block_shape_k * 2
        elif block_shape_m <= 32:
            warp_shape_k = warp_shape_k // 2

    min_warp_shape_n = 32 if layer_config.a_dtype.num_bits == 16 else 16
    # Keep a complete four-warp WGMMA group while fitting output width.
    while layer_config.shape_n % block_shape_n != 0:
        block_shape_n //= 2
        assert block_shape_n >= min_warp_shape_n * 4
    warp_shape_n = min(warp_shape_n, block_shape_n // 4)

    # Earlier shape fitting can reduce block K below the initial warp K.
    warp_shape_k = min(warp_shape_k, block_shape_k)
    while layer_config.shape_k % block_shape_k != 0:
        block_shape_k = block_shape_k // 2
        warp_shape_k = min(warp_shape_k, block_shape_k)
        assert block_shape_k >= warp_shape_k

    if layer_config.use_packed_k_layout:
        warp_shape_k = 128
        block_shape_k = max(block_shape_k, warp_shape_k)

    if problem.gemm_type == GemmType.INDEXED and layer_config.use_packed_k_layout:
        while block_shape_n // warp_shape_n * (block_shape_k // warp_shape_k) > 8:
            block_shape_k //= 2

    # Extra K partitions inflate the producer and dequantization state without
    # enough reduction work to amortize them. Preserve the long-K schedules.
    needs_smaller_k_tile = layer_config.shape_k <= 2048 and block_shape_k > 64
    dense_small_wna16 = (
        problem.gemm_type == GemmType.DENSE
        and layer_config.a_dtype.num_bits == 16
        and layer_config.b_dtype.num_bits < 16
        and (layer_config.b_dtype.num_bits == 4 or needs_smaller_k_tile)
        and problem.shape_m <= 128
        and layer_config.shape_n % 128 == 0
        and layer_config.shape_k % 64 == 0
    )
    if dense_small_wna16:
        block_shape_n = 128
        block_shape_k = 64
        warp_shape_n = 32
        warp_shape_k = 64
    if problem.use_batch_invariant:
        # Keep one K partition and the same reduction tile for every batch size.
        batch_invariant_k = 128 if layer_config.use_packed_k_layout else 1024 // layer_config.a_dtype.num_bits
        block_shape_k = min(batch_invariant_k, layer_config.shape_k & -layer_config.shape_k)
        warp_shape_k = block_shape_k

    config = {
        "block_shape": (block_shape_m, block_shape_n, block_shape_k),
        "warp_shape": (block_shape_m, warp_shape_n, warp_shape_k),
        "use_stream_k": not problem.use_batch_invariant,
        "use_f16_accum": problem.use_f16_accum,
        "num_stages": 4,
    }

    can_increase_residency = problem.gemm_type == GemmType.INDEXED and not layer_config.use_packed_k_layout
    if can_increase_residency and block_shape_n == 64 and block_shape_m <= 32:
        config["num_ctas_per_sm"] = 2

    if problem.gemm_type != GemmType.INDEXED:
        config["use_warp_spec"] = True
        config["use_tma"] = True
        if dense_small_wna16:
            config["num_ctas_per_sm"] = 2

        if (
            layer_config.shape_n % (block_shape_n * 2) == 0
            and problem.shape_m / block_shape_m >= 4
            and problem.gemm_type == GemmType.DENSE
        ):
            config["multi_cast_size_a"] = 2

    return config


def select_grouped_scale(
    problem: TuningProblem,
) -> TuningDecision:
    layer_config = problem.layer_config
    use_wgmma_ss = layer_config.a_dtype.num_bits == layer_config.b_dtype.num_bits
    if problem.use_f16_accum:
        max_block_m = 256
    elif layer_config.input_scale_group_size > 0:
        max_block_m = 160
    elif layer_config.weight_scale_group_size < 128:
        max_block_m = 192
    else:
        max_block_m = 200
    block_shape_m = select_sm90_block_m(
        layer_config,
        problem.shape_m,
        max_block_m,
        problem.gemm_type,
        problem.expert_probability_cv,
    )
    if problem.use_batch_invariant:
        block_ks = (min(128, layer_config.shape_k & -layer_config.shape_k),)
    else:
        block_ks = (256, 128, 64) if block_shape_m <= 32 else (128, 64)
    use_multicast = problem.gemm_type == GemmType.DENSE and problem.shape_m / block_shape_m >= 4

    candidates = []
    # Candidate order records measured preference; legality supplies fallbacks.
    tile_shapes = ((128, 32), (128, 16), (64, 16)) if use_wgmma_ss else ((128, 32), (64, 16))
    for block_shape_n, warp_shape_n in tile_shapes:
        for block_shape_k in block_ks:
            multicast_values = (True, False) if use_multicast else (False,)
            for multicast in multicast_values:
                config = {
                    "block_shape": (
                        block_shape_m,
                        block_shape_n,
                        block_shape_k,
                    ),
                    "warp_shape": (
                        block_shape_m,
                        warp_shape_n,
                        min(128, block_shape_k),
                    ),
                    "use_stream_k": not problem.use_batch_invariant,
                    "use_f16_accum": problem.use_f16_accum,
                    "num_stages": 4,
                }
                if problem.gemm_type != GemmType.INDEXED:
                    config["use_warp_spec"] = True
                    config["use_tma"] = True
                if multicast:
                    config["multi_cast_size_a"] = 2
                candidate = ScheduleCandidate.from_config(
                    "grouped_scale_"
                    f"n{block_shape_n}_wn{warp_shape_n}_k{block_shape_k}_"
                    f"{'multicast' if multicast else 'direct'}",
                    config,
                )
                candidate = ScheduleCandidate.from_config(
                    candidate.candidate_id,
                    fit_sm90_tile(
                        layer_config,
                        candidate.to_config(),
                        problem.use_f16_accum,
                        gemm_type=problem.gemm_type,
                        use_m_major_input_scale=problem.use_m_major_input_scale,
                    ),
                )
                candidates.append(fit_pipeline_stages(problem, candidate))

    has_routing_grid = problem.gemm_type != GemmType.DENSE and problem.device.num_sms is not None
    if has_routing_grid:
        policy = Sm90CandidatePolicy()
        analyses = tuple(_analyze_moe_candidate(problem, candidate, policy) for candidate in candidates)
    else:
        analyses = tuple(analyze_candidate(problem, candidate) for candidate in candidates)
    legal = [analysis for analysis in analyses if analysis.legal]
    selected = legal[0] if legal else None
    if selected is None:
        rejected = {analysis.candidate.candidate_id: analysis.rejection_reasons for analysis in analyses}
        raise AssertionError(f"no legal grouped-scale SM90 schedule: {rejected}")

    reason = "selected the first legal measured-priority candidate"
    if has_routing_grid:
        assert problem.device.num_sms is not None
        selected = min(
            legal,
            key=lambda analysis: (analysis.waves or math.inf) * math.prod(analysis.candidate.block_shape[:2]),
        )
        resident_ctas = problem.device.num_sms * selected.candidate.get("num_ctas_per_sm", 1)
        if selected.num_output_tiles >= 16 * resident_ctas:
            candidate = selected.candidate.with_updates(use_stream_k=False)
            final_analysis = analyze_candidate(problem, candidate)
            analyses = tuple(final_analysis if analysis is selected else analysis for analysis in analyses)
            selected = final_analysis
        reason = "selected a legal MoE schedule using routed tiles, residency and padded work per wave"

    return TuningDecision(
        problem=problem,
        family="grouped_scale",
        selected=selected.candidate,
        considered=analyses,
        reason=reason,
    )


def get_scaled_a8_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    use_wgmma_ss = layer_config.a_dtype.num_bits == layer_config.b_dtype.num_bits
    has_native_w8a8 = layer_config.a_dtype.num_bits == 8 and layer_config.a_dtype == layer_config.b_dtype
    has_unpacked_weights = (
        layer_config.a_dtype.num_bits == 8
        and not use_wgmma_ss
        and not layer_config.use_packed_k_layout
        and not layer_config.is_group_input_scale
        and layer_config.shape_k > 512
    )
    if not (has_native_w8a8 or has_unpacked_weights):
        return None
    if use_f16_accum or use_batch_invariant:
        return None
    if gemm_type != GemmType.DENSE and not layer_config.num_experts:
        return None
    is_grouped = gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED)
    has_grouped_block_scale = is_grouped and layer_config.is_block_weight_scale
    if has_grouped_block_scale and is_h20:
        return None
    if layer_config.shape_n % 128 or layer_config.shape_k % 128:
        return None
    if get_problem_rejection_reasons(layer_config, MmaType.WGMMA):
        return None

    num_sms = current_device.sm_count
    alignment = 16 if layer_config.a_dtype.is_integer_type else 8
    short_k = layer_config.shape_k <= 512

    def make_config(block_m, block_n, warp_n, use_warp_spec, use_stream_k, stages=4, ctas=1, ring=True):
        use_ring = use_warp_spec and ring
        if has_unpacked_weights:
            warp_n = max(32, warp_n)
        return {
            "mma_type": MmaType.WGMMA.value,
            "block_shape": (block_m, block_n, 128),
            "warp_shape": (block_m, warp_n, 128),
            "num_stages": stages,
            "num_ctas_per_sm": ctas,
            "use_tma": use_warp_spec,
            "use_warp_spec": use_warp_spec,
            "use_stream_k": bool(use_stream_k),
            "smem_reuse_mode": "none" if use_ring else "all_stages",
            "producer_stage_unroll": 2 if use_ring else None,
            "raster_group_m": 1,
        }

    def fits_resources(config):
        fitted = fit_sm90_tile(
            layer_config,
            config,
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if fitted != config:
            return False
        smem_size = estimate_smem_size_layer(
            layer_config,
            config["block_shape"],
            gemm_type,
            config["num_stages"],
            warp_shape=config["warp_shape"],
            smem_reuse_mode=config["smem_reuse_mode"],
            use_tma=config["use_tma"],
            use_warp_spec=config["use_warp_spec"],
            mma_type=MmaType.WGMMA,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        return smem_size * config["num_ctas_per_sm"] <= 227 * 1024

    if gemm_type == GemmType.DENSE:
        if short_k:
            return None
        n_tiles = layer_config.shape_n // 128
        has_wide_short_grid = layer_config.shape_k <= 1024 and n_tiles >= num_sms / 4 and shape_m <= 128
        if is_h20 and shape_m < 32:
            return None
        if has_wide_short_grid and shape_m >= 32:
            m_tiles = max(1, math.ceil(num_sms * 2 / 3 / n_tiles))
            block_m = min(64, math.ceil(shape_m / m_tiles / alignment) * alignment)
            config = make_config(block_m, 128, 16, True, False)
        elif has_wide_short_grid:
            block_m = int(shape_m * n_tiles / num_sms) // alignment * alignment
            block_m = min(64, max(16, block_m))
            config = make_config(block_m, 128, 16, False, False, stages=3, ctas=2)
        elif is_h20:
            if shape_m < 32 or (layer_config.shape_k >= 4096 and n_tiles < num_sms / 4):
                return None
            m_tiles = max(1, math.ceil(num_sms * 2 / 3 / n_tiles))
            block_m = min(64, math.ceil(shape_m / m_tiles / alignment) * alignment)
            if math.ceil(shape_m / block_m) * n_tiles < num_sms * 2 / 3:
                return None
            config = make_config(block_m, 128, 16, True, False)
        else:
            if shape_m <= 64:
                return None
            wide_tiles = math.ceil(shape_m / 64) * (layer_config.shape_n // 256)
            if layer_config.shape_n % 256 == 0 and wide_tiles >= num_sms * 2 / 3:
                config = make_config(64, 256, 32, True, False)
            else:
                config = make_config(64, 128, 32, True, True, ring=False)
        return config if fits_resources(config) else None

    counts = DeviceHeuristics.get_expert_rows(layer_config, shape_m, gemm_type, expert_probability_cv)
    is_indexed = gemm_type == GemmType.INDEXED
    if is_h20 and is_indexed and not has_unpacked_weights and layer_config.shape_k < 4096:
        full_tiles = (counts // 32).sum(axis=1) * (layer_config.shape_n // 256)
        if short_k or full_tiles.mean() < num_sms:
            return None
    candidates = []
    max_block_m = 64 if is_h20 else 80
    if is_h20 and is_indexed and not has_unpacked_weights:
        max_block_m = 32
    for block_m in range(8, max_block_m + 1, 8):
        if layer_config.a_dtype.is_integer_type and block_m > 32 and block_m % 16:
            continue
        m_tiles = ((counts + block_m - 1) // block_m).sum(axis=1)
        wide_tiles = m_tiles * (layer_config.shape_n // 256)
        use_wide_tile = layer_config.shape_n % 256 == 0 and wide_tiles.mean() >= num_sms * 2 / 3
        block_n = 256 if use_wide_tile else 128
        tiles = m_tiles * (layer_config.shape_n // block_n)
        config = make_config(block_m, block_n, block_n // 8, True, True, stages=3 if short_k else 4)
        if fits_resources(config):
            score = float(np.ceil(tiles / num_sms).mean() * (block_m + 32) * block_n / 128)
            candidates.append((score, config))
    if not candidates:
        return None
    config = min(candidates, key=lambda item: item[0])[1]
    block_m, block_n, _ = config["block_shape"]
    if is_h20 and is_indexed and not has_unpacked_weights and block_n == 256 and block_m <= 16:
        full_tiles = (counts // block_m).sum(axis=1) * (layer_config.shape_n // block_n)
        if full_tiles.mean() < num_sms:
            return None
    if has_unpacked_weights and (is_grouped or is_indexed) and block_m <= 32:
        synchronous = make_config(block_m, block_n, block_n // 8, False, True, stages=3, ctas=2)
        synchronous["wgmma_split_issue_wait"] = True
        can_widen_tile = not is_indexed or block_m <= 16
        if is_h20 and can_widen_tile and layer_config.shape_n % 512 == 0:
            wider = make_config(block_m, 512, 64, False, True, stages=3, ctas=2)
            wider["wgmma_split_issue_wait"] = True
            if fits_resources(wider):
                synchronous = wider
        if fits_resources(synchronous):
            block_n = synchronous["block_shape"][1]
            tiles = ((counts + block_m - 1) // block_m).sum(axis=1) * (layer_config.shape_n // block_n)
            has_short_indexed_grid = is_h20 and is_indexed and tiles.mean() < num_sms * 4
            if has_short_indexed_grid and layer_config.shape_k <= 4096:
                return None
            if not is_h20 and tiles.mean() < num_sms:
                return config
            three_ctas = synchronous | {"num_ctas_per_sm": 3}
            warp_m, warp_n, warp_k = synchronous["warp_shape"]
            math_threads = math.prod(synchronous["block_shape"]) // math.prod(synchronous["warp_shape"]) * 32
            register_budget = 65536 // (math_threads * 3) // 8 * 8
            operand_registers = warp_m * warp_k // 128
            accumulator_registers = warp_m * warp_n // 32
            if not layer_config.use_fused_e8m0_scale and layer_config.is_group_weight_scale:
                accumulator_registers *= 2
            has_register_headroom = operand_registers + accumulator_registers + 32 <= register_budget
            two_cta_work = np.ceil(tiles / (num_sms * 2)).mean() * 2
            three_cta_work = np.ceil(tiles / (num_sms * 3)).mean() * 3
            if has_register_headroom and three_cta_work < two_cta_work and fits_resources(three_ctas):
                synchronous = three_ctas
            has_short_grid = tiles.mean() < num_sms * 2
            if is_indexed and not has_short_grid:
                overlap = config | {"num_ctas_per_sm": 2, "producer_stage_unroll": 1}
                overlap["wgmma_split_issue_wait"] = True
                if is_h20:
                    if block_m > 16 and fits_resources(overlap):
                        return overlap
                else:
                    single_cta_work = np.ceil(tiles / num_sms).mean()
                    if two_cta_work > single_cta_work * 1.1:
                        return config
                    has_short_reduction = layer_config.shape_k <= 4096
                    if block_m > 16 and has_short_reduction and has_register_headroom:
                        if fits_resources(three_ctas):
                            return three_ctas
                    if fits_resources(overlap):
                        return overlap
            return synchronous
    return config


def get_packed_wna8_config(
    layer_config: LayerConfig,
    shape_m: int,
    gemm_type: GemmType,
    use_f16_accum: bool,
    use_batch_invariant: bool,
    *,
    is_h20: bool = False,
    use_m_major_input_scale: bool = False,
    expert_probability_cv: float = DeviceHeuristics.expert_probability_cv,
) -> dict | None:
    """Select packed weight tiles from routing work and SM90 resources."""
    if not layer_config.use_packed_k_layout:
        return None
    if is_h20 and gemm_type == GemmType.DENSE and not layer_config.use_fused_e8m0_scale:
        return None
    if use_f16_accum or use_batch_invariant:
        return None
    if layer_config.shape_n % 128 or layer_config.shape_k % 128:
        return None

    if gemm_type == GemmType.DENSE:
        # Use one K warp to limit register pressure on small grids,
        # and fit M to the rows to reduce padding.
        block_m = math.ceil(shape_m / math.ceil(shape_m / 64))
        alignment = 16 if layer_config.a_dtype.is_integer_type else 8
        block_m = math.ceil(block_m / alignment) * alignment
        tiles = math.ceil(shape_m / block_m) * (layer_config.shape_n // 128)
        can_split_k = not is_h20 and shape_m >= 32 and layer_config.shape_k >= 2048
        if can_split_k and tiles < current_device.sm_count * 2 / 3:
            return {
                "mma_type": MmaType.WGMMA.value,
                "block_shape": (block_m, 128, 128),
                "warp_shape": (block_m, 16, 128),
                "num_stages": 4,
                "num_ctas_per_sm": 1,
                "use_tma": True,
                "use_warp_spec": True,
                "use_stream_k": True,
                "wgmma_split_issue_wait": True,
            }
        if shape_m <= 64:
            return None

    if gemm_type == GemmType.DENSE:
        counts = np.asarray([[shape_m]])
    else:
        counts = DeviceHeuristics.get_expert_rows(layer_config, shape_m, gemm_type, expert_probability_cv)

    has_indexed_weight_groups = (
        gemm_type == GemmType.INDEXED
        and layer_config.is_group_weight_scale
        and not layer_config.use_fused_e8m0_scale
    )
    max_block_m = 64 if is_h20 else 160
    num_sms = current_device.sm_count
    best_config = None
    best_score = math.inf
    large_configs = []

    def estimate_work(config, expert_rows):
        block_m, block_n, _ = config["block_shape"]
        resident_ctas = config["num_ctas_per_sm"]
        m_tiles = ((expert_rows + block_m - 1) // block_m).sum(axis=1)
        tiles = m_tiles * (layer_config.shape_n // block_n)
        waves = np.ceil(tiles / (num_sms * resident_ctas))
        return float(waves.mean() * resident_ctas * (block_m + 32) * block_n / 128)

    for block_m in range(8, max_block_m + 1, 8):
        if layer_config.a_dtype.is_integer_type and block_m % 16:
            continue
        is_small_tile = block_m <= 32
        block_n = 256 if is_small_tile and layer_config.shape_n % 256 == 0 else 128
        warp_n = 32 if block_n == 256 else 16
        can_use_warp_spec = gemm_type != GemmType.INDEXED or has_indexed_weight_groups
        use_warp_spec = not is_h20 and not is_small_tile and can_use_warp_spec
        can_overlap_indexed = not is_h20 or 16 < block_m <= 32
        use_indexed_overlap = gemm_type == GemmType.INDEXED and can_overlap_indexed and block_m <= 64
        use_warp_spec |= use_indexed_overlap
        resident_ctas = 1 if use_warp_spec and not use_indexed_overlap else 2
        stages = 3 if is_small_tile and not use_warp_spec else (5 if is_h20 else 4)
        if use_indexed_overlap:
            stages = 4
        config = {
            "mma_type": MmaType.WGMMA.value,
            "block_shape": (block_m, block_n, 128),
            "warp_shape": (block_m, warp_n, 128),
            "num_stages": stages,
            "num_ctas_per_sm": resident_ctas,
            "use_tma": use_warp_spec,
            "use_warp_spec": use_warp_spec,
            "use_stream_k": gemm_type != GemmType.DENSE,
            "wgmma_use_late_as": layer_config.is_group_input_scale,
            "wgmma_split_issue_wait": not (use_warp_spec and has_indexed_weight_groups),
            "smem_reuse_mode": "none" if use_warp_spec else "all_stages",
            "raster_group_m": 1,
        }
        if use_warp_spec:
            config["producer_stage_unroll"] = 1 if is_h20 and use_indexed_overlap else 2
        fitted = fit_sm90_tile(
            layer_config,
            config,
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if fitted != config:
            continue
        smem_size = estimate_smem_size_layer(
            layer_config,
            config["block_shape"],
            gemm_type,
            stages,
            warp_shape=config["warp_shape"],
            smem_reuse_mode=config["smem_reuse_mode"],
            use_tma=use_warp_spec,
            use_warp_spec=use_warp_spec,
            mma_type=MmaType.WGMMA,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if smem_size * resident_ctas > 227 * 1024:
            continue

        # Balance padded math against repeated weight loading and dequantization.
        # Round each sampled expert before accounting for incomplete waves.
        score = estimate_work(config, counts)
        if block_m >= 128:
            large_configs.append((config, score))
        if score < best_score:
            best_score = score
            best_config = config
    has_routed_rows = gemm_type in (GemmType.GROUPED_CONTIGUOUS, GemmType.INDEXED)
    if has_routed_rows and best_config is not None and best_config["block_shape"][0] > 128:
        # Prefer narrower RS tiles when nominal work is close and skewed work improves.
        # Rank by nominal work to preserve gains near tile boundaries.
        skewed_rows = DeviceHeuristics._sample_expert_rows(
            shape_m, layer_config.num_experts, max(1.0, expert_probability_cv)
        )
        skewed_work = estimate_work(best_config, skewed_rows)
        safer_configs = [
            (config, score)
            for config, score in large_configs
            if config["block_shape"][0] < best_config["block_shape"][0]
            and score <= best_score * 1.1
            and estimate_work(config, skewed_rows) < skewed_work
        ]
        if safer_configs:
            best_config = min(safer_configs, key=lambda item: item[1])[0]

    if best_config is not None:
        block_m, block_n, _ = best_config["block_shape"]
        # Preserve the existing Stream-K decode schedules when little work can
        # amortize a full packed stage or the indexed row-gathering setup.
        if is_h20 and gemm_type == GemmType.INDEXED and block_m <= 16:
            return None
        if is_h20 and block_m == 8 and block_n == 256:
            return None
        if gemm_type == GemmType.DENSE or not layer_config.use_fused_e8m0_scale:
            m_tiles = ((counts + block_m - 1) // block_m).sum(axis=1)
            tiles = m_tiles * (layer_config.shape_n // block_n)
            resident_ctas = 1 if layer_config.use_fused_e8m0_scale else best_config["num_ctas_per_sm"]
            if tiles.mean() < num_sms * resident_ctas / 2:
                return None
    return best_config


def _moe_ctas_per_sm(
    problem: TuningProblem,
    analysis: CandidateAnalysis,
    policy: Sm90CandidatePolicy,
) -> int:
    # Thread caps stand in for the measured register launch-bound cliffs.
    resource_limit = 1
    if (
        analysis.num_threads <= policy.max_indexed_threads_for_two_ctas
        and analysis.smem_size * 2 <= problem.device.resident_smem_size
    ):
        resource_limit = 2
    block_shape = analysis.candidate.block_shape
    if (
        problem.layer_config.a_dtype.num_bits == 16
        and problem.layer_config.b_dtype.num_bits == 4
        and block_shape[0] == 8
        and analysis.num_threads <= policy.max_indexed_threads_for_three_ctas
        and analysis.smem_size * 3 <= problem.device.resident_smem_size
    ):
        resource_limit = 3

    resource_limit = min(resource_limit, analysis.thread_smem_cta_limit)
    if problem.device.num_sms is None:
        raise ValueError("MoE selection requires a device SM count")
    grid_limit = math.ceil(analysis.num_output_tiles / problem.device.num_sms)
    return max(1, min(resource_limit, grid_limit))


def _analyze_moe_candidate(
    problem: TuningProblem,
    candidate: ScheduleCandidate,
    policy: Sm90CandidatePolicy,
) -> CandidateAnalysis:
    analysis = analyze_candidate(problem, candidate)
    if not analysis.legal:
        return analysis
    candidate = candidate.with_updates(
        num_ctas_per_sm=_moe_ctas_per_sm(
            problem,
            analysis,
            policy,
        )
    )
    fitted = fit_sm90_tile(
        problem.layer_config,
        candidate.to_config(),
        problem.use_f16_accum,
        gemm_type=problem.gemm_type,
        use_m_major_input_scale=problem.use_m_major_input_scale,
    )
    candidate = ScheduleCandidate.from_config(candidate.candidate_id, fitted)
    return analyze_candidate(problem, candidate)


def _half_k_candidate(
    problem: TuningProblem,
    source: ScheduleCandidate,
) -> ScheduleCandidate | None:
    block_shape = source.block_shape
    warp_shape = source.warp_shape
    smaller_block_k = block_shape[2] // 2
    scale_groups_align = all(
        not group_size or group_size % smaller_block_k == 0 or smaller_block_k % group_size == 0
        for group_size in (
            problem.layer_config.input_scale_group_size,
            problem.layer_config.weight_scale_group_size,
        )
    )
    if block_shape[2] < warp_shape[2] * 2 or not scale_groups_align:
        return None

    config = source.to_config()
    config.pop("num_ctas_per_sm", None)
    return ScheduleCandidate.from_config(
        "indexed_a16_half_k",
        config,
    ).with_updates(
        block_shape=(*block_shape[:2], smaller_block_k),
        warp_shape=(
            *warp_shape[:2],
            min(warp_shape[2], smaller_block_k),
        ),
    )


def _split_n_widen_k_candidate(
    problem: TuningProblem,
    source: ScheduleCandidate,
    *,
    candidate_id: str,
) -> ScheduleCandidate | None:
    block_shape = source.block_shape
    if not (
        problem.layer_config.a_dtype.num_bits == 16
        and problem.layer_config.b_dtype.num_bits == 4
        and block_shape[1] >= 256
        and block_shape[2] == 64
    ):
        return None

    config = source.to_config()
    config.pop("num_ctas_per_sm", None)
    return ScheduleCandidate.from_config(candidate_id, config).with_updates(
        block_shape=(
            block_shape[0],
            block_shape[1] // 2,
            block_shape[2] * 2,
        ),
    )


def select_moe_a16(
    problem: TuningProblem,
    policy: Sm90CandidatePolicy,
) -> TuningDecision:
    if problem.device.num_sms is None:
        raise ValueError("MoE selection requires a device SM count")
    base = fit_pipeline_stages(
        problem,
        ScheduleCandidate.from_config(
            "indexed_a16_base",
            build_sm90_seed_config(problem),
        ),
    )
    base_option = _MoeA16Option(base, "base", 0)
    options = [base_option]
    mma_option = None
    layer = problem.layer_config
    can_use_mma = layer.b_dtype.num_bits < 16 and not problem.use_f16_accum
    has_short_aligned_k = layer.shape_k <= 1024 and layer.shape_k % 64 == 0
    if can_use_mma and has_short_aligned_k and layer.shape_n % 128 == 0:
        mma = ScheduleCandidate.from_config(
            "moe_a16_mma",
            {
                "mma_type": MmaType.MMA.value,
                "block_shape": (16, 128, 64),
                "warp_shape": (16, 32, 32),
                "num_stages": 4,
                "num_ctas_per_sm": 1,
                "use_stream_k": False,
                "use_warp_spec": False,
                "use_tma": False,
            },
        )
        mma_option = _MoeA16Option(mma, "mma", 3)
        options.append(mma_option)
    half_k = _half_k_candidate(problem, base)
    half_option = None
    if half_k is not None:
        half_option = _MoeA16Option(half_k, "half_k", 1, base_option)
        options.append(half_option)
    for source, candidate_id in (
        (base_option, "indexed_a16_split_n_widen_k_from_base"),
        (half_option, "indexed_a16_split_n_widen_k"),
    ):
        if source is None:
            continue
        split = _split_n_widen_k_candidate(
            problem,
            source.candidate,
            candidate_id=candidate_id,
        )
        if split is not None:
            options.append(_MoeA16Option(split, "split_n_widen_k", 2, source))

    analyses = {
        option: _analyze_moe_candidate(
            problem,
            option.candidate,
            policy,
        )
        for option in options
    }
    base_analysis = analyses[base_option]
    if not base_analysis.legal:
        raise AssertionError(base_analysis.rejection_reasons)
    eligible_reasons = {
        base_option: "selected the base MoE A16 schedule",
    }

    if half_option is not None:
        half_analysis = analyses[half_option]
        if (
            base_analysis.candidate.num_ctas_per_sm == 1
            and half_analysis.legal
            and half_analysis.candidate.num_ctas_per_sm > base_analysis.candidate.num_ctas_per_sm
        ):
            eligible_reasons[half_option] = "halved K because it increased CTA residency"

    for option in options:
        if option.transform != "split_n_widen_k":
            continue
        assert option.parent is not None
        parent_reason = eligible_reasons.get(option.parent)
        analysis = analyses[option]
        parent_analysis = analyses[option.parent]
        if (
            parent_reason is not None
            and analysis.legal
            and analysis.candidate.num_ctas_per_sm > parent_analysis.candidate.num_ctas_per_sm
            and analysis.waves is not None
            and parent_analysis.waves is not None
            and analysis.waves <= parent_analysis.waves
        ):
            eligible_reasons[option] = f"{parent_reason}; split N and widened K without adding a grid wave"
    selected = max(
        eligible_reasons,
        key=lambda option: option.priority,
    )

    if mma_option is not None:
        mma_analysis = analyses[mma_option]
        selected_analysis = analyses[selected]
        mma_block_m = mma_analysis.candidate.block_shape[0]
        selected_block_m = selected_analysis.candidate.block_shape[0]
        mma_rows = problem.estimate_num_blocks_m(mma_block_m) * mma_block_m
        selected_rows = problem.estimate_num_blocks_m(selected_block_m) * selected_block_m
        has_fewer_waves = (
            mma_analysis.waves is not None
            and selected_analysis.waves is not None
            and mma_analysis.waves < selected_analysis.waves
        )
        if mma_analysis.legal and has_fewer_waves and mma_rows <= selected_rows:
            selected = mma_option
            eligible_reasons[selected] = "selected short-K MMA with fewer waves and no extra row padding"

    final_candidate = analyses[selected].candidate.with_updates(
        use_stream_k=(
            bool(analyses[selected].candidate.get("use_stream_k", True))
            and analyses[selected].num_output_tiles < problem.device.num_sms
        )
    )
    final_analysis = analyze_candidate(problem, final_candidate)
    if not final_analysis.legal:
        raise AssertionError(final_analysis.rejection_reasons)
    return TuningDecision(
        problem=problem,
        family="indexed_a16" if problem.gemm_type == GemmType.INDEXED else "grouped_a16",
        selected=final_candidate,
        considered=tuple(final_analysis if option is selected else analyses[option] for option in options),
        reason=eligible_reasons[selected],
    )

import math

from humming import dtypes
from humming.config import GemmType, MmaType, TuningConfig
from humming.config.mma import get_default_mma_type, get_register_budget_error
from humming.device import current_device
from humming.tune.raster import get_moe_raster_group_m
from humming.tune.sm8x import Sm89Heuristics
from humming.utils.math import round_up
from humming.utils.smem import estimate_smem_size_layer


class Sm120Heuristics(Sm89Heuristics):
    sm_version: int = 120
    max_smem_size: int = 99 * 1024
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
        dtypes.float8e3m4,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.float4e2m1, dtypes.float4e0m3]

    @staticmethod
    def _has_complete_warpgroups(config) -> bool:
        num_math_warps = math.prod(config["block_shape"]) // math.prod(config["warp_shape"])
        return num_math_warps % 4 == 0

    @classmethod
    def should_use_pdl_for_input(cls, layer_config, shape_m: int) -> bool:
        return layer_config.shape_n >= 4096 and shape_m <= 32

    @classmethod
    def _fit_dense_block_m_to_grid(
        cls,
        block_shape: tuple[int, int],
        layer_config,
        shape_m: int,
        gemm_type: GemmType,
        num_ctas_per_sm: int,
    ) -> int:
        block_shape_m, block_shape_n = block_shape
        mma_type = get_default_mma_type(layer_config)
        unsupported = gemm_type != GemmType.DENSE or mma_type not in (MmaType.MMA, MmaType.MXMMA)
        unsuitable_m = block_shape_m <= 16 or shape_m <= block_shape_m <= 32
        if unsupported or unsuitable_m:
            return block_shape_m

        min_grid_blocks = math.ceil(current_device.sm_count * num_ctas_per_sm / 2)
        num_blocks_n = layer_config.shape_n // block_shape_n
        num_blocks_m = math.ceil(shape_m / block_shape_m)
        num_blocks = num_blocks_n * num_blocks_m
        if num_blocks >= min_grid_blocks:
            return block_shape_m

        max_num_blocks_m = num_blocks_m * 4
        target_grid_blocks = min_grid_blocks if num_blocks * 4 >= min_grid_blocks else num_blocks * 2
        candidates = [
            candidate
            for candidate in range(16, block_shape_m, 16)
            if num_blocks_n * math.ceil(shape_m / candidate) >= target_grid_blocks
            and math.ceil(shape_m / candidate) <= max_num_blocks_m
        ]
        return min(
            candidates,
            key=lambda candidate: (round_up(shape_m, candidate), -candidate),
            default=block_shape_m,
        )

    @classmethod
    def _has_group_fp_scales(cls, a_dtype, group_size, use_fused_e8m0_scale) -> bool:
        return (
            a_dtype.is_floating_point_type
            and a_dtype.num_bits <= 8
            and group_size > 0
            and not use_fused_e8m0_scale
        )

    @classmethod
    def get_base_config(
        cls,
        a_dtype: dtypes.DataType,
        b_dtype: dtypes.DataType,
        group_size: int,
        use_f16_accum: bool,
        use_fused_e8m0_scale: bool,
        gemm_type: GemmType,
        shape_k: int,
    ):
        if cls._has_group_fp_scales(a_dtype, group_size, use_fused_e8m0_scale):
            block_k = 512 // a_dtype.num_bits
            if a_dtype.num_bits == 8 and b_dtype.num_bits < a_dtype.num_bits:
                return {
                    "block_shape": (112, 256, block_k),
                    "warp_shape": (112, 32, block_k),
                    "num_stages": 2,
                }
            return {
                "block_shape": (256, 128, block_k),
                "warp_shape": (128, 32, block_k),
                "num_stages": 2,
            }
        if a_dtype.is_floating_point_type and a_dtype.num_bits == 16 and not use_f16_accum:
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (128, 32, 64),
                "num_stages": 2,
            }
        return super().get_base_config(
            a_dtype, b_dtype, group_size, use_f16_accum, use_fused_e8m0_scale, gemm_type, shape_k
        )

    @classmethod
    def get_config(
        cls,
        layer_config,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        config = super().get_config(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
            use_m_major_input_scale,
        )
        if use_batch_invariant:
            return config

        a = layer_config.a_dtype
        is_wna16 = a.is_floating_point_type and a.num_bits == 16 and not use_f16_accum
        if a.is_floating_point_type and a.num_bits <= 8 and not layer_config.use_fused_e8m0_scale:
            config["use_tma"] = True
            config["use_warp_spec"] = cls._has_complete_warpgroups(config)
        elif is_wna16:
            config["use_tma"] = True
            config["use_warp_spec"] = cls._has_complete_warpgroups(config)
            config["num_stages"] = cls._fit_num_stages(
                layer_config,
                config,
                gemm_type,
                smem_reuse_mode="all_stages",
            )

        if gemm_type == GemmType.INDEXED:
            config["use_tma_a"] = False
            config["use_tma_c"] = False
            config["use_warp_spec"] = False

        if gemm_type != GemmType.DENSE and is_wna16 and layer_config.b_dtype.num_bits <= 8:
            tokens_per_expert = shape_m / layer_config.num_experts
            if tokens_per_expert >= 96:
                cls._use_m_tile(layer_config, config, gemm_type, 128, shape_m)
            elif tokens_per_expert >= 48:
                cls._use_m_tile(layer_config, config, gemm_type, 64, shape_m)

        num_b_bits = layer_config.b_dtype.num_bits
        block_m = config["block_shape"][0]
        if gemm_type == GemmType.GROUPED_MASKED and is_wna16 and num_b_bits <= 8 and block_m <= 64:
            cls._use_tma_b_only(config)

        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        has_group_fp_scales = cls._has_group_fp_scales(
            layer_config.a_dtype, group_size, layer_config.use_fused_e8m0_scale
        )
        is_mxmma = has_group_fp_scales and get_default_mma_type(layer_config) == MmaType.MXMMA
        if gemm_type != GemmType.INDEXED and has_group_fp_scales:
            num_stages = cls._fit_num_stages(layer_config, config, gemm_type, smem_reuse_mode="last_stage")
            config["num_stages"] = num_stages
            config["smem_reuse_mode"] = "last_stage"

        if gemm_type == GemmType.DENSE:
            num_blocks_n = layer_config.shape_n // config["block_shape"][1]
            num_blocks_k = layer_config.shape_k // config["block_shape"][2]
            num_blocks_nk = num_blocks_n * num_blocks_k
            config.pop("num_sms", None)

            is_small_m = shape_m <= config["block_shape"][0] <= 32
            if is_small_m and num_blocks_nk < current_device.sm_count * 3:
                config["num_stages"] = 3

            if is_small_m and num_blocks_nk < current_device.sm_count * 2:
                config["num_stages"] = 2
                config["use_warp_spec"] = False
                config["use_tma"] = False

            cls._tune_small_m_dense(layer_config, config, shape_m, is_wna16)
            cls._tune_mxmma_dense(layer_config, config, shape_m, is_mxmma, num_blocks_nk)
            config["use_stream_k"] = cls._should_use_stream_k(layer_config, config, shape_m)
            use_stream_k = config["use_stream_k"]
            rebalance_stream_k = cls._should_rebalance_stream_k(layer_config, config, shape_m)
            rebalance_stream_k = rebalance_stream_k and use_stream_k
            block_m, block_n, block_k = config["block_shape"]
            num_output_tiles = math.ceil(shape_m / block_m) * math.ceil(layer_config.shape_n / block_n)
            num_sms = current_device.sm_count
            if rebalance_stream_k and block_m <= 16 and num_output_tiles * 3 >= num_sms:
                config["use_stream_k"] = False

            moderate_nk_grid = cls.sm_version == 120 and num_sms * 2 <= num_blocks_nk < num_sms * 3
            if is_small_m and not config["use_stream_k"] and moderate_nk_grid:
                config["use_tma"] = False
                config["use_warp_spec"] = False
                for key in [key for key in config if key.startswith("use_tma_")]:
                    config.pop(key)
                warp_m, warp_n, warp_k = config["warp_shape"]
                config["warp_shape"] = (warp_m, warp_n, min(block_k, warp_k * 2))

            if config["use_stream_k"] and block_m <= 16:
                warp_m, warp_n, warp_k = config["warp_shape"]
                mn_warps = block_m // warp_m * (block_n // warp_n)
                target_k_warps = max(1, 4 // mn_warps)
                config["warp_shape"] = (warp_m, warp_n, max(warp_k, block_k // target_k_warps))
                if not cls._has_complete_warpgroups(config):
                    config["use_warp_spec"] = False

            cls._rebalance_dense_warps(layer_config, config, shape_m)
            if has_group_fp_scales:
                config["num_stages"] = min(
                    config["num_stages"],
                    cls._fit_num_stages(
                        layer_config,
                        config,
                        gemm_type,
                        smem_reuse_mode=config.get("smem_reuse_mode", "all_stages"),
                    ),
                )

        cls._tune_scaled_mma_pipeline(layer_config, config, gemm_type, use_m_major_input_scale)
        cls._tune_small_tile_transfers(layer_config, config, gemm_type, shape_m)
        if is_wna16:
            cls._tune_moe_w4a16(layer_config, config, gemm_type, shape_m)
        cls._tune_short_moe_residency(layer_config, config, gemm_type, shape_m, use_m_major_input_scale)
        previous_block_m = config["block_shape"][0]
        cls._tune_moe_tiles(layer_config, config, gemm_type, shape_m, use_m_major_input_scale)
        should_align_scales = not config.get("use_warp_spec", False)
        should_align_scales |= config["block_shape"][0] > previous_block_m
        if should_align_scales:
            cls._align_moe_scale_tiles(layer_config, config, gemm_type, use_m_major_input_scale)
        is_fp8 = a in (dtypes.float8e4m3, dtypes.float8e5m2) and a == layer_config.b_dtype
        use_grouped_raster = cls.sm_version == 120 and gemm_type == GemmType.GROUPED_CONTIGUOUS
        if use_grouped_raster and is_fp8 and layer_config.is_block_weight_scale:
            config["raster_group_m"] = get_moe_raster_group_m(
                shape_m, layer_config.num_experts, config["block_shape"][0], group_m=4, min_m_tiles=2
            )
        tokens_per_expert = shape_m / (layer_config.num_experts or 1)
        config["evict_weights_first"] = tokens_per_expert <= config["block_shape"][0]
        if gemm_type != GemmType.DENSE:
            config["evict_weights_first"] = tokens_per_expert <= config["block_shape"][0] / 2
        return config

    @classmethod
    def _tune_short_moe_residency(
        cls, layer_config, config, gemm_type, shape_m, use_m_major_input_scale
    ) -> None:
        if gemm_type != GemmType.INDEXED or get_default_mma_type(layer_config) != MmaType.MXMMA:
            return
        block_m, block_n, block_k = config["block_shape"]
        warp_k = config["warp_shape"][2]
        if block_m > 32 or config.get("use_stream_k", False):
            return
        if config.get("num_ctas_per_sm", 1) != 1:
            return
        if cls.sm_version == 121 and layer_config.a_dtype.num_bits == 8:
            return

        candidate_k = max(warp_k, 4 * layer_config.input_scale_group_size)
        max_k_iters = 2 if layer_config.a_dtype.num_bits == 4 else 4
        if layer_config.shape_k > max_k_iters * candidate_k or layer_config.shape_k % candidate_k:
            return
        candidate_n = min(256, block_n * (block_k // warp_k))
        if candidate_n < 128 or layer_config.shape_n % candidate_n:
            return
        m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, block_m, gemm_type)
        if m_tiles * (layer_config.shape_n // candidate_n) < 2 * current_device.sm_count:
            return

        # Move intra-CTA K parallelism into independently scheduled four-warp CTAs.
        candidate = config | {
            "block_shape": (block_m, candidate_n, candidate_k),
            "warp_shape": (block_m, candidate_n // 4, candidate_k),
            "num_ctas_per_sm": 2,
            "num_stages": 2,
            "use_tma": True,
            "use_tma_a": False,
            "use_tma_c": False,
            "use_tma_as": False,
            "use_tma_as2": False,
            "use_warp_spec": False,
            "smem_reuse_mode": "all_stages",
        }
        if cls._fit_moe_tile_pipeline(layer_config, candidate, gemm_type, use_m_major_input_scale):
            config.update(candidate)

    @classmethod
    def _align_moe_scale_tiles(cls, layer_config, config, gemm_type, use_m_major_input_scale) -> None:
        if gemm_type == GemmType.DENSE or get_default_mma_type(layer_config) != MmaType.MXMMA:
            return
        is_mixed_input = layer_config.a_dtype.num_bits > layer_config.b_dtype.num_bits
        if not is_mixed_input or not layer_config.is_group_input_scale:
            return

        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        scale_word_k = 4 * layer_config.input_scale_group_size
        if warp_k >= scale_word_k:
            return
        aligned_block_k = scale_word_k * (block_k // warp_k)
        if layer_config.shape_k % aligned_block_k:
            return

        candidate = config | {
            "block_shape": (block_m, block_n, aligned_block_k),
            "warp_shape": (warp_m, warp_n, scale_word_k),
        }
        if cls._fit_moe_tile_pipeline(layer_config, candidate, gemm_type, use_m_major_input_scale):
            config.update(candidate)

    @classmethod
    def _tune_moe_tiles(cls, layer_config, config, gemm_type, shape_m, use_m_major_input_scale) -> None:
        if gemm_type not in (GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS):
            return
        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        mma_type = get_default_mma_type(layer_config)
        is_mxmma = mma_type == MmaType.MXMMA
        raster_block_m, raster_block_n = block_m, block_n
        has_large_experts = block_m >= 64 and shape_m >= layer_config.num_experts * 128
        if is_mxmma:
            if shape_m < layer_config.num_experts * block_m:
                return
        elif not has_large_experts:
            return
        if config.get("num_ctas_per_sm", 1) != 1 or config.get("use_f16_accum", False):
            return

        num_math_warps = math.prod(config["block_shape"]) // math.prod(config["warp_shape"])
        has_split_k = block_k != warp_k
        if has_split_k:
            if not is_mxmma:
                return
            # Move K-parallel warps to N so larger M tiles avoid a partial-output reduction.
            block_n *= block_k // warp_k
            block_k = warp_k
            if block_n > 256 or layer_config.shape_n % block_n:
                return
        base_config = config | {"block_shape": (block_m, block_n, block_k)}
        base_configs = [base_config]
        is_native_fp4 = is_mxmma and layer_config.a_dtype.num_bits == 4
        is_mixed_mxmma = is_mxmma and layer_config.a_dtype.num_bits > layer_config.b_dtype.num_bits
        if is_native_fp4:
            candidate = config | {
                "block_shape": (128, 256, 128),
                "warp_shape": (64, 64, 128),
                "num_stages": 4,
            }
            is_aligned = layer_config.shape_n % 256 == 0 and layer_config.shape_k % 128 == 0
            m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, 128, gemm_type)
            has_many_waves = m_tiles * (layer_config.shape_n // 256) >= 4 * current_device.sm_count
            if is_aligned and has_many_waves:
                if cls.sm_version == 120:
                    candidate.update(num_sms=current_device.sm_count, use_stream_k=False)
                if cls._fit_moe_tile_pipeline(layer_config, candidate, gemm_type, use_m_major_input_scale):
                    base_configs = [candidate] if cls.sm_version == 120 else [*base_configs, candidate]

        can_narrow_mixed_tile = layer_config.shape_n % 128 == 0 and layer_config.shape_k % 128 == 0
        if is_mixed_mxmma and block_n > 128 and can_narrow_mixed_tile:
            weight_tile_bytes = block_n * layer_config.shape_k * layer_config.b_dtype.num_bits // 8
            resident_ctas = config.get("num_sms", current_device.sm_count)
            weight_bytes = layer_config.shape_n * layer_config.shape_k * layer_config.b_dtype.num_bits // 8
            has_large_expert_weights = 2 * weight_bytes > current_device.l2_cache_size
            if has_large_expert_weights and weight_tile_bytes * resident_ctas > current_device.l2_cache_size:
                base_configs.append(base_config | {"block_shape": (block_m, 128, 128)})

        max_accumulators = 128 if is_native_fp4 else 96
        if mma_type == MmaType.MMA and layer_config.a_dtype.num_bits == 8:
            max_accumulators = 64
        original_m = config["block_shape"][0]
        original_m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, original_m, gemm_type)
        max_padded_rows = original_m_tiles * original_m * 1.05
        best_config = config
        best_cost = cls._estimate_moe_tile_cost(layer_config, config, gemm_type, shape_m) * 0.98
        for base_config in base_configs:
            _, block_n, block_k = base_config["block_shape"]
            reference_m, reference_n, _ = best_config["block_shape"]
            reference_m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, reference_m, gemm_type)
            m_warp_choices = [config["block_shape"][0] // warp_m]
            if is_mxmma:
                m_warp_choices.extend(count for count in (1, 2) if count not in m_warp_choices)
            max_stages = base_config["num_stages"]
            if is_mxmma:
                max_stages = min(max(4, max_stages), max(2, layer_config.shape_k // block_k))
            has_narrower_n = block_n < base_configs[0]["block_shape"][1]
            max_tile_m = 160 if is_mixed_mxmma and has_narrower_n else 144
            min_m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, max_tile_m, gemm_type)
            use_indexed_warp_spec = has_narrower_n and gemm_type == GemmType.INDEXED
            use_indexed_warp_spec &= min_m_tiles > 1.25 * layer_config.num_experts
            for tile_m in range(64, max_tile_m + 1, 16):
                m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, tile_m, gemm_type)
                for num_m_warps in m_warp_choices:
                    if not is_mxmma and tile_m % (16 * num_m_warps):
                        num_m_warps = 1
                    if num_math_warps % num_m_warps or tile_m % (16 * num_m_warps):
                        continue
                    num_n_warps = num_math_warps // num_m_warps
                    if block_n % num_n_warps:
                        continue
                    candidate_warp_m = tile_m // num_m_warps
                    candidate_warp_n = block_n // num_n_warps
                    if candidate_warp_n % 8:
                        continue
                    accumulators = candidate_warp_m * candidate_warp_n // 32
                    if accumulators > max_accumulators:
                        continue
                    candidate = base_config | {
                        "block_shape": (tile_m, block_n, block_k),
                        "warp_shape": (candidate_warp_m, candidate_warp_n, block_k),
                        "num_stages": max_stages,
                    }
                    if use_indexed_warp_spec:
                        candidate["use_warp_spec"] = True
                    if not cls._fit_moe_tile_pipeline(
                        layer_config, candidate, gemm_type, use_m_major_input_scale
                    ):
                        continue

                    if num_m_warps == 1 and m_tiles * tile_m > max_padded_rows:
                        continue
                    if m_tiles * (layer_config.shape_n // block_n) < current_device.sm_count * 2:
                        continue
                    if has_narrower_n:
                        min_tile_m = original_m if use_indexed_warp_spec else reference_m
                        if tile_m * block_n < min_tile_m * reference_n:
                            continue
                        if m_tiles * 1.1 >= reference_m_tiles:
                            continue
                    has_repeated_weights = m_tiles >= 2 * layer_config.num_experts
                    has_wider_n = block_n > base_configs[0]["block_shape"][1]
                    if has_repeated_weights and has_wider_n:
                        weight_tile_bytes = block_n * layer_config.shape_k * layer_config.b_dtype.num_bits / 8
                        resident_ctas = candidate.get("num_sms", current_device.sm_count)
                        resident_ctas *= candidate.get("num_ctas_per_sm", 1)
                        if weight_tile_bytes * resident_ctas > current_device.l2_cache_size:
                            continue
                    needs_weight_reuse = is_mixed_mxmma and candidate.get("use_warp_spec", False)
                    needs_weight_reuse &= has_repeated_weights and not has_narrower_n
                    if needs_weight_reuse:
                        working_set = cls._estimate_moe_weight_working_set(layer_config, candidate, m_tiles)
                        if working_set > current_device.l2_cache_size:
                            continue
                    cost = cls._estimate_moe_tile_cost(layer_config, candidate, gemm_type, shape_m)
                    if cost < best_cost:
                        best_cost, best_config = cost, candidate

        config.update(best_config)
        if is_native_fp4 and config.get("use_warp_spec", False) and config["num_stages"] == 2:
            candidate = config | {"use_warp_spec": False, "smem_reuse_mode": "all_stages", "num_stages": 3}
            if cls._fit_moe_tile_pipeline(layer_config, candidate, gemm_type, use_m_major_input_scale):
                if candidate["num_stages"] > config["num_stages"]:
                    config.update(candidate)

        use_indexed_raster = cls.sm_version == 121 and gemm_type == GemmType.INDEXED
        use_indexed_raster &= has_large_experts and not has_split_k
        if use_indexed_raster:
            block_m, block_n, _ = config["block_shape"]
            if is_mixed_mxmma and block_n < raster_block_n:
                block_m, block_n = raster_block_m, raster_block_n
            weight_tile_bytes = block_n * layer_config.shape_k * layer_config.b_dtype.num_bits // 8
            num_sms = config.get("num_sms", current_device.sm_count)
            # Keep adjacent M tiles close when concurrent weight reads exceed L2 capacity.
            if weight_tile_bytes * num_sms > current_device.l2_cache_size:
                config["raster_group_m"] = get_moe_raster_group_m(
                    shape_m, layer_config.num_experts, block_m, group_m=2
                )

    @classmethod
    def _estimate_moe_weight_working_set(cls, layer_config, config, m_tiles) -> float:
        n_tiles = layer_config.shape_n // config["block_shape"][1]
        resident_ctas = config.get("num_sms", current_device.sm_count)
        resident_ctas *= config.get("num_ctas_per_sm", 1)
        concurrent_experts = 1 + resident_ctas * layer_config.num_experts / (m_tiles * n_tiles)
        concurrent_experts = min(layer_config.num_experts, concurrent_experts)
        weight_bytes = layer_config.shape_n * layer_config.shape_k * layer_config.b_dtype.num_bits / 8
        return concurrent_experts * weight_bytes

    @classmethod
    def _estimate_moe_tile_cost(cls, layer_config, config, gemm_type, shape_m) -> float:
        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, _ = config["warp_shape"]
        m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, block_m, gemm_type)
        # Balance padded rows against reloading B, with headroom for the pipeline and operands.
        weight_rows = block_n * layer_config.b_dtype.num_bits / layer_config.a_dtype.num_bits
        is_mixed_mxmma = get_default_mma_type(layer_config) == MmaType.MXMMA
        is_mixed_mxmma &= layer_config.a_dtype.num_bits > layer_config.b_dtype.num_bits
        if is_mixed_mxmma and m_tiles >= 2 * layer_config.num_experts:
            working_set = cls._estimate_moe_weight_working_set(layer_config, config, m_tiles)
            cache_pressure = working_set / current_device.l2_cache_size
            weight_rows *= max(1, cache_pressure)
        tile_rows = block_m + weight_rows
        if is_mixed_mxmma and gemm_type == GemmType.INDEXED and config.get("use_warp_spec", False):
            # Producer warps overlap indexed activation gathers with the consumers.
            tile_rows = max(block_m, weight_rows)
        cost = m_tiles * tile_rows / block_n
        if is_mixed_mxmma:
            scale_word_k = 4 * layer_config.input_scale_group_size
            cost *= max(1, scale_word_k / block_k)
        accumulators = warp_m * warp_n // 32
        return cost * (1 + 0.5 / (config["num_stages"] - 1)) * (1 + accumulators / 512)

    @classmethod
    def _fit_moe_tile_pipeline(cls, layer_config, config, gemm_type, use_m_major_input_scale) -> bool:
        mma_type = get_default_mma_type(layer_config)
        is_mixed_mxmma = mma_type == MmaType.MXMMA
        is_mixed_mxmma &= layer_config.a_dtype.num_bits > layer_config.b_dtype.num_bits
        is_native_fp4 = mma_type == MmaType.MXMMA and layer_config.a_dtype.num_bits == 4
        needs_operand_budget = is_mixed_mxmma or is_native_fp4
        if needs_operand_budget and not config.get("use_warp_spec", False):
            warp_m, warp_n, warp_k = config["warp_shape"]
            warp_k = max(warp_k, 4 * layer_config.input_scale_group_size)
            math_threads = math.prod(config["block_shape"]) // math.prod(config["warp_shape"]) * 32
            resident_threads = math_threads * config.get("num_ctas_per_sm", 1)
            register_budget = min(255, current_device.max_registers_per_sm // resident_threads)
            activation_registers = warp_m * warp_k * layer_config.a_dtype.num_bits / 1024
            operand_bits = max(layer_config.a_dtype.num_bits, layer_config.b_dtype.num_bits)
            weight_registers = warp_n * warp_k * operand_bits / 1024
            accumulator_registers = warp_m * warp_n / 32
            address_registers = 32
            if is_native_fp4:
                # Packed activation staging adds pressure to wide FP4 M warps.
                activation_registers *= 2
                address_registers = 16
            register_demand = accumulator_registers + activation_registers + weight_registers
            if register_demand + address_registers > register_budget:
                return False

        tuning_config = TuningConfig(
            block_shape=config["block_shape"],
            warp_shape=config["warp_shape"],
            num_ctas_per_sm=config.get("num_ctas_per_sm", 1),
            use_warp_spec=config.get("use_warp_spec", False),
        )
        if get_register_budget_error(layer_config, tuning_config):
            return False
        for num_stages in range(config["num_stages"], 1, -1):
            smem = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                gemm_type,
                num_stages,
                mma_type=get_default_mma_type(layer_config),
                warp_shape=config["warp_shape"],
                smem_reuse_mode=config.get("smem_reuse_mode", "all_stages"),
                use_tma=config.get("use_tma", False),
                use_warp_spec=config.get("use_warp_spec", False),
                use_tma_c=config.get("use_tma_c", config.get("use_tma", False)),
                raster_group_m=config.get("raster_group_m", 1),
                use_m_major_input_scale=use_m_major_input_scale,
                output_chunk_rows=config.get("output_chunk_rows", 0),
            )
            if smem * config.get("num_ctas_per_sm", 1) <= cls.max_smem_size:
                config["num_stages"] = num_stages
                return True
        return False

    @classmethod
    def _tune_moe_w4a16(cls, layer_config, config, gemm_type, shape_m) -> None:
        if gemm_type not in (GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS):
            return
        if layer_config.b_dtype.num_bits != 4:
            return

        tokens_per_expert = shape_m / layer_config.num_experts
        block_m = next((tile for tile in (8, 16, 32, 48) if tokens_per_expert < tile * 0.9), None)
        if block_m is None:
            return

        max_ctas = 4 if block_m <= 16 else 2
        best_ctas = 1
        best_config = None
        for block_n, block_k in ((128, 64), (64, 128)):
            if layer_config.shape_n % block_n or layer_config.shape_k % block_k:
                continue
            candidate = config | {
                "block_shape": (block_m, block_n, block_k),
                "warp_shape": (block_m, 64, 32),
                "num_stages": 4,
                "num_sms": current_device.sm_count,
                "use_stream_k": True,
                "output_chunk_rows": 0,
            }
            cls._use_cp_async(candidate)
            smem = estimate_smem_size_layer(
                layer_config,
                candidate["block_shape"],
                gemm_type,
                candidate["num_stages"],
                warp_shape=candidate["warp_shape"],
            )
            grid_ctas = max(1, shape_m * (layer_config.shape_n // block_n) * 4 // current_device.sm_count)
            num_ctas = min(max_ctas, cls.max_smem_size // smem, grid_ctas)
            if num_ctas > best_ctas:
                candidate["num_ctas_per_sm"] = num_ctas
                best_ctas = num_ctas
                best_config = candidate

        if best_config is not None:
            config.clear()
            config.update(best_config)

    @staticmethod
    def _use_cp_async(config) -> None:
        config.update(use_tma=False, use_warp_spec=False)
        for key in [key for key in config if key.startswith("use_tma_")]:
            config.pop(key)

    @classmethod
    def _tune_small_tile_transfers(cls, layer_config, config, gemm_type, shape_m) -> None:
        block_m, block_n, block_k = config["block_shape"]
        num_a_bits = layer_config.a_dtype.num_bits
        num_b_bits = layer_config.b_dtype.num_bits
        if gemm_type == GemmType.DENSE and shape_m <= block_m <= 16:
            has_dequant_mainloop = num_a_bits < 16 and num_a_bits != num_b_bits
            if not config["use_stream_k"] or not has_dequant_mainloop:
                cls._use_cp_async(config)
        elif gemm_type == GemmType.INDEXED:
            has_small_weight_stage = block_m <= 16 and block_n * block_k * num_b_bits <= 32768
            has_low_bit_weights = num_a_bits == 8 and num_b_bits <= 4 and block_m <= 64
            if has_small_weight_stage or has_low_bit_weights:
                cls._use_cp_async(config)

    @classmethod
    def _tune_scaled_mma_pipeline(cls, layer_config, config, gemm_type, use_m_major_input_scale) -> None:
        mma_type = get_default_mma_type(layer_config)
        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        original_stages = config["num_stages"]
        is_mxfp8 = mma_type == MmaType.MXMMA and layer_config.a_dtype.num_bits == 8
        is_mxfp8 &= layer_config.b_dtype.num_bits == 8
        is_mxfp8 &= layer_config.is_group_input_scale or layer_config.is_group_weight_scale
        should_refit = False
        has_small_tile = block_m <= 32
        can_use_scale_tma = use_m_major_input_scale and config.get("use_tma", False)
        keep_scale_tma = can_use_scale_tma and gemm_type != GemmType.INDEXED and has_small_tile
        if is_mxfp8 and not keep_scale_tma:
            has_irregular_moe_tile = gemm_type != GemmType.DENSE and block_m < 64
            if has_small_tile or has_irregular_moe_tile:
                block_m = min(block_m, 32)
                prefer_wide_tile = cls.sm_version == 120 and block_m >= 32 and gemm_type != GemmType.INDEXED
                max_block_n = 128 if gemm_type == GemmType.DENSE or prefer_wide_tile else 64
                block_n = min(block_n, max_block_n)
                block_k = 256 if block_n == 64 else 128
                warp_m, warp_n = 16, 32
                k_warps = max(1, 4 // ((block_m // warp_m) * (block_n // warp_n)))
                warp_k = block_k // k_warps
                cls._use_cp_async(config)
                config["num_stages"] = 3
                should_refit = True
            elif gemm_type == GemmType.INDEXED:
                block_m, block_n, block_k = 64, min(block_n, 128), 128
                warp_m, warp_n, warp_k = 32, 32, 128
                cls._use_cp_async(config)
                config["num_stages"] = 3
                should_refit = True
            elif block_k > warp_k:
                block_m, block_n, block_k = 64, min(128, layer_config.shape_n), 64
                warp_m, warp_n, warp_k = 32, 32, 64
                config.update(use_tma=True, use_warp_spec=True, num_stages=4)
                should_refit = True

        has_software_group_scales = mma_type == MmaType.MMA and layer_config.a_dtype.num_bits == 8
        has_software_group_scales &= layer_config.is_group_input_scale or layer_config.is_group_weight_scale
        if has_software_group_scales and warp_m > 64:
            block_m, block_n, block_k = 64, min(block_n, 128), min(block_k, 128)
            warp_m, warp_n, warp_k = 32, min(warp_n, 32), block_k
            cls._use_cp_async(config)
            config["num_stages"] = min(config["num_stages"], 3)
            should_refit = True

        if should_refit:
            while layer_config.shape_n % block_n:
                block_n //= 2
            while layer_config.shape_k % block_k:
                block_k //= 2
                warp_k = min(warp_k, block_k)
            has_same_block = tuple(config["block_shape"]) == (block_m, block_n, block_k)
            has_same_warp = tuple(config["warp_shape"]) == (warp_m, warp_n, warp_k)
            if has_same_block and has_same_warp:
                config["num_stages"] = min(config["num_stages"], original_stages)
            config.update(block_shape=(block_m, block_n, block_k), warp_shape=(warp_m, warp_n, warp_k))
            reuse_mode = config.get("smem_reuse_mode", "all_stages")
            fitted_stages = cls._fit_num_stages(layer_config, config, gemm_type, reuse_mode)
            config["num_stages"] = min(config["num_stages"], fitted_stages)
            for num_stages in range(config["num_stages"], 1, -1):
                smem = estimate_smem_size_layer(
                    layer_config,
                    config["block_shape"],
                    gemm_type,
                    num_stages,
                    warp_shape=config["warp_shape"],
                    smem_reuse_mode=reuse_mode,
                    use_tma=config.get("use_tma", False),
                    use_warp_spec=config.get("use_warp_spec", False),
                    output_chunk_rows=config.get("output_chunk_rows", 0),
                )
                config["num_stages"] = num_stages
                if smem * config["num_ctas_per_sm"] <= cls.max_smem_size:
                    break
            config["num_ctas_per_sm"] = min(config["num_ctas_per_sm"], cls.max_smem_size // smem)

        has_long_warp_k = warp_m >= 32 and warp_k >= 256 and not config.get("use_warp_spec", False)
        if has_long_warp_k and (is_mxfp8 or has_software_group_scales):
            config.update(producer_stage_unroll=2, consumer_stage_unroll=2)

    @classmethod
    def _tune_small_m_dense(cls, layer_config, config, shape_m: int, is_wna16: bool) -> None:
        block_m, block_n, block_k = config["block_shape"]
        if shape_m > block_m or block_m > 32 or not is_wna16:
            return

        use_stream_k = config.get("use_stream_k", False)
        shape_k = layer_config.shape_k

        if config.get("use_tma", False):
            cls._use_tma_b_only(config)
        elif not use_stream_k and block_k == 128 and shape_k % (block_k * 2) == 0:
            warp_m, warp_n, _ = config["warp_shape"]
            config["block_shape"] = (block_m, block_n, block_k * 2)
            config["warp_shape"] = (warp_m, warp_n, 64)
            config["num_stages"] = 3

    @classmethod
    def _tune_mxmma_dense(cls, layer_config, config, shape_m, is_mxmma, num_blocks_nk) -> None:
        block_m, block_n, _ = config["block_shape"]
        num_b_bits = layer_config.b_dtype.num_bits
        num_a_bits = layer_config.a_dtype.num_bits
        if not is_mxmma or num_a_bits != 8 or num_b_bits >= 8:
            return

        shape_n = layer_config.shape_n
        shape_k = layer_config.shape_k
        if shape_m < 128 or block_n >= 256 or shape_n % 256 or shape_k % 64:
            return

        block_m = max(block_m, 96)
        config["block_shape"] = (block_m, 256, 64)
        config["warp_shape"] = (block_m, 32, 64)
        config["num_stages"] = 4 if num_blocks_nk >= current_device.sm_count * 3 else 2

    @classmethod
    def _should_use_stream_k(cls, layer_config, config, shape_m: int) -> bool:
        return config.get("use_stream_k", False)

    @classmethod
    def _should_rebalance_stream_k(cls, layer_config, config, shape_m: int) -> bool:
        block_m, block_n, block_k = config["block_shape"]
        num_blocks_m = math.ceil(shape_m / block_m)
        num_blocks_n = math.ceil(layer_config.shape_n / block_n)
        num_k_tiles = layer_config.shape_k // block_k
        num_ctas = current_device.sm_count * config.get("num_ctas_per_sm", 1)
        return num_blocks_m * num_blocks_n * num_k_tiles < num_ctas * config["num_stages"] * 2

    @classmethod
    def _rebalance_dense_warps(cls, layer_config, config, shape_m: int) -> None:
        use_stream_k = config.get("use_stream_k", False)
        block_m, block_n, _ = config["block_shape"]
        _, warp_n, _ = config["warp_shape"]
        mma_k = 1024 // layer_config.a_dtype.num_bits
        block_k = mma_k * 2
        while block_k > layer_config.shape_k:
            block_k //= 2
        if layer_config.shape_k % block_k:
            return

        num_warps_n = block_n // warp_n
        num_warps_m = min(2, block_m // 16, max(1, 4 // num_warps_n))
        if num_warps_m * num_warps_n < 4:
            return

        warp_m = block_m // num_warps_m
        if warp_m % 16:
            return

        num_k_tiles = layer_config.shape_k // block_k
        avoid_small_warp_m = cls.sm_version == 120 and shape_m <= block_m and warp_m < 32 and num_k_tiles > 4
        if not use_stream_k and avoid_small_warp_m:
            return

        candidate = config | {
            "block_shape": (block_m, block_n, block_k),
            "warp_shape": (warp_m, warp_n, block_k),
        }
        if all(candidate[key] == config[key] for key in ("block_shape", "warp_shape")):
            return

        if warp_m < 64:
            candidate["use_tma"] = False
            candidate["use_warp_spec"] = False
            for key in [key for key in candidate if key.startswith("use_tma_")]:
                candidate.pop(key)

        max_stages = min(4, candidate["num_stages"])
        for num_stages in range(max_stages, 1, -1):
            smem = estimate_smem_size_layer(
                layer_config,
                candidate["block_shape"],
                GemmType.DENSE,
                num_stages,
                warp_shape=candidate["warp_shape"],
                smem_reuse_mode=candidate.get("smem_reuse_mode", "all_stages"),
                use_tma=candidate.get("use_tma", False),
                use_warp_spec=candidate.get("use_warp_spec", False),
                output_chunk_rows=candidate.get("output_chunk_rows", 0),
            )
            if smem <= cls.max_smem_size:
                candidate["num_stages"] = num_stages
                config.clear()
                config.update(candidate)
                return

    @staticmethod
    def _use_tma_b_only(config) -> None:
        config.update(
            use_tma=True,
            use_warp_spec=False,
            use_tma_a=False,
            use_tma_as=False,
            use_tma_as2=False,
            use_tma_b=True,
            use_tma_c=False,
            use_tma_bs=False,
            use_tma_bs2=False,
            use_tma_bzp=False,
            use_tma_bias=False,
        )

    @classmethod
    def _use_m_tile(cls, layer_config, config, gemm_type, block_m: int, shape_m: int) -> None:
        _, block_n, block_k = config["block_shape"]
        _, warp_n, warp_k = config["warp_shape"]
        config["block_shape"] = (block_m, block_n, block_k)
        config["warp_shape"] = (block_m, warp_n, warp_k)
        config["use_warp_spec"] = config.get("use_warp_spec", False) and cls._has_complete_warpgroups(config)
        config["num_stages"] = cls._fit_num_stages(
            layer_config,
            config,
            gemm_type,
            smem_reuse_mode="all_stages",
        )

        m_tiles = cls.estimate_num_blocks_m(layer_config, shape_m, block_m)
        min_grid_blocks = math.ceil(current_device.sm_count * 2 / 3)
        has_sparse_grid = m_tiles * (layer_config.shape_n // block_n) < min_grid_blocks
        if block_n <= 64 or not has_sparse_grid:
            return

        # A larger M tile can leave too few output tiles and consume the
        # shared memory needed for pipelining. Narrow N only if both improve.
        block_n //= 2
        warp_n = min(warp_n, 32)
        k_warps = max(1, 8 // (block_n // warp_n))
        warp_k = max(32, block_k // k_warps)
        candidate = config | {
            "block_shape": (block_m, block_n, block_k),
            "warp_shape": (block_m, warp_n, warp_k),
            "num_sms": current_device.sm_count,
        }
        candidate["use_warp_spec"] &= cls._has_complete_warpgroups(candidate)
        candidate["num_stages"] = cls._fit_num_stages(
            layer_config, candidate, gemm_type, smem_reuse_mode="all_stages"
        )
        if candidate["num_stages"] > config["num_stages"]:
            config.update(candidate)

    @classmethod
    def _fit_num_stages(cls, layer_config, config, gemm_type, smem_reuse_mode: str) -> int:
        best = None
        for num_stages in range(2, 6 if cls.sm_version == 121 else 5):
            smem = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                gemm_type,
                num_stages,
                warp_shape=config["warp_shape"],
                smem_reuse_mode=smem_reuse_mode,
                use_tma=config.get("use_tma", False),
                use_warp_spec=config["use_warp_spec"],
                output_chunk_rows=config.get("output_chunk_rows", 0),
            )
            if smem <= cls.max_smem_size:
                best = num_stages
        if best is None:
            raise ValueError(f"No pipeline stage count fits shared memory on SM{cls.sm_version}: {config}")
        return best

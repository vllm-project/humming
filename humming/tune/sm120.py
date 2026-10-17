import math

from humming import dtypes
from humming.config import GemmType, MmaType
from humming.config.mma import get_default_mma_type
from humming.device import current_device
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
        # A single M tile reads each weight tile once; keep the stream from evicting reused data in L2.
        # For MoE, use the average rows per expert.
        config["evict_weights_first"] = shape_m / (layer_config.num_experts or 1) <= config["block_shape"][0]
        return config

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

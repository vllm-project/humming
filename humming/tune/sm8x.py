import math

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import current_device
from humming.tune.base import DeviceHeuristics
from humming.utils.math import ceil_div
from humming.utils.smem import estimate_smem_size_layer


class Sm80Heuristics(DeviceHeuristics):
    max_smem_size: int = 163 * 1024
    sm_version: int = 80
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.int4]

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
        has_raw_weight = a_dtype.num_bits == b_dtype.num_bits
        if a_dtype.num_bits == 16 or group_size == 0 and has_raw_weight:
            if use_f16_accum:
                return {
                    "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                    "warp_shape": (128, 64, 512 // a_dtype.num_bits),
                }
            else:
                return {
                    "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                    "warp_shape": (64, 64, 1024 // a_dtype.num_bits),
                }
        elif group_size == 0:
            return {
                "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
            }
        elif group_size > 0:
            return {
                "block_shape": (64, 128, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
                "num_ctas_per_sm": 2,
            }

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        config = super().get_config(
            layer_config,
            shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if cls._should_use_decode_tile(layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type):
            config = cls._get_decode_tile_config(config, layer_config, shape_m)

        config["evict_weights_first"] = cls._evict_weights_first(config, layer_config, shape_m, gemm_type)
        return config

    @staticmethod
    def _evict_weights_first(
        config: dict, layer_config: LayerConfig, shape_m: int, gemm_type: GemmType
    ) -> bool:
        tokens_per_expert = shape_m / (layer_config.num_experts or 1)
        block_m = config["block_shape"][0]
        if gemm_type != GemmType.DENSE:
            return tokens_per_expert <= block_m * 0.5 <= 8
        return tokens_per_expert <= block_m

    @classmethod
    def _should_use_decode_tile(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType,
    ) -> bool:
        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        shape_k = layer_config.shape_k
        is_dense_w4a16 = gemm_type == GemmType.DENSE and layer_config.a_dtype.num_bits == 16
        is_dense_w4a16 &= layer_config.b_dtype.num_bits == 4 and group_size == 128
        is_decode_size = (shape_m <= 16 and shape_k <= 5120) or (shape_m <= 32 and shape_k <= 4096)
        is_tile_aligned = layer_config.shape_n % 64 == 0 and shape_k % 128 == 0
        uses_default_math = not use_f16_accum and not use_batch_invariant
        is_sm80 = cls.sm_version == 80
        return is_sm80 and is_dense_w4a16 and is_decode_size and is_tile_aligned and uses_default_math

    @classmethod
    def _get_decode_tile_config(cls, config: dict, layer_config: LayerConfig, shape_m: int) -> dict:
        """Give each CTA one whole output tile for decode-size W4A16 GEMMs.

        Stream-K splits a tile's K range across CTAs, and every split adds a
        pipeline refill and a global-memory reduction: about 5 us on A100,
        while the weights of a 4096x4096 layer stream in about 6 us. A
        data-parallel grid of one wave is faster even when some SMs idle.
        Layers with K > 5120 keep Stream-K: their tiles are long enough to
        hide the reduction, and the data-parallel grid lost there.
        """
        num_sms = current_device.sm_count
        num_ctas_per_sm = 2
        max_num_tiles = num_sms * num_ctas_per_sm
        # At M <= 8 a tile is cheap, so fill every SM; at larger M a 128-column
        # tile pays off once it covers 55% of the SMs (56 tiles lost at N=7168).
        min_num_tiles = num_sms if shape_m <= 8 else math.ceil(0.55 * num_sms)

        shape_n = layer_config.shape_n
        one_wave_block_n = [
            block_n for block_n in (128, 64) if shape_n % block_n == 0 and shape_n // block_n <= max_num_tiles
        ]
        filling_block_n = [block_n for block_n in one_wave_block_n if shape_n // block_n >= min_num_tiles]
        if filling_block_n:
            block_n = filling_block_n[0]
        elif one_wave_block_n and shape_m <= 16:
            # Up to M = 16 a 64-column grid wins even when it fills under half the SMs.
            block_n = one_wave_block_n[-1]
        else:
            return config

        block_m = 8 if shape_m <= 8 else 16 if shape_m <= 16 else 32
        warp_n = 64
        block_k = 128
        warp_k = block_k * (block_n // warp_n) // 4
        block_shape = (block_m, block_n, block_k)
        warp_shape = (block_m, warp_n, warp_k)
        num_stages = 6 if block_n == 64 else 4

        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            GemmType.DENSE,
            num_stages,
            warp_shape=warp_shape,
            output_chunk_rows=config.get("output_chunk_rows", 0),
        )
        if smem_size * num_ctas_per_sm > cls.max_smem_size:
            return config

        config = dict(config)
        config.update(
            block_shape=block_shape,
            warp_shape=warp_shape,
            use_stream_k=False,
            num_stages=num_stages,
            num_ctas_per_sm=num_ctas_per_sm,
            num_sms=ceil_div(shape_n // block_n, num_ctas_per_sm),
        )
        return config


class Sm86Heuristics(DeviceHeuristics):
    max_smem_size: int = 99 * 1024
    sm_version: int = 86
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8]
    b4_allowed_dtypes: list[dtypes.DataType] = [dtypes.int4]

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
        if a_dtype.num_bits == 16 and use_f16_accum:
            num_stages = 3 if b_dtype.num_bits < 8 else 2
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (64, 64, 64),
                "num_stages": num_stages,
            }
        elif a_dtype.num_bits == 16 and not use_f16_accum:
            num_stages = 3 if b_dtype.num_bits < 8 else 2
            return {
                "block_shape": (128, 256, 64),
                "warp_shape": (64, 64, 64),
                "num_stages": num_stages,
            }
        elif use_fused_e8m0_scale:
            return {
                "block_shape": (128, 128, 512 // a_dtype.num_bits),
                "warp_shape": (128, 32, 512 // a_dtype.num_bits),
                "num_stages": 2,
                "num_warps_per_sm": 2,
            }
        elif group_size == 0 and a_dtype.num_bits == b_dtype.num_bits:
            return {
                "block_shape": (128, 256, 512 // a_dtype.num_bits),
                "warp_shape": (64, 64, 512 // a_dtype.num_bits),
                "num_stages": 2,
            }
        elif group_size == 0:
            return {
                "block_shape": (128, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (128, 32, 1024 // a_dtype.num_bits),
                "num_stages": 2,
            }
        else:
            return {
                "block_shape": (64, 256, 1024 // a_dtype.num_bits),
                "warp_shape": (64, 32, 1024 // a_dtype.num_bits),
                "num_stages": 2,
            }


class Sm87Heuristics(Sm80Heuristics):
    """Jetson AGX Orin: SM80's tiles, except for 16-bit-activation NVFP4 MoE, whose expert GEMMs are
    bound by the 204.8 GB/s LPDDR5 on 16 SMs. There, measured tiles replace SM80's choice up to 64
    tokens per expert."""

    sm_version: int = 87

    # (max tokens per expert, block shape, warp shape, stages, CTAs per SM), in increasing order. The
    # ranges between 1.125 and 7.125 tokens per expert and above 64 keep SM80's tiles (no consistent
    # gain measured there).
    _NVFP4_MOE_TILES = (
        (1.125, (8, 128, 128), (8, 32, 64), 4, 2),
        (7.125, None, None, 0, 0),
        (14.375, (16, 256, 64), (16, 64, 64), 4, 2),
        (29.75, (32, 256, 64), (32, 64, 64), 4, 2),
        (64.0, (64, 256, 64), (64, 32, 64), 4, 1),
    )

    @classmethod
    def _get_nvfp4_moe_tile(cls, layer_config: LayerConfig, shape_m: int, gemm_type: GemmType) -> dict | None:
        is_nvfp4 = layer_config.b_dtype == dtypes.float4e2m1 and layer_config.weight_scale_group_size == 16
        is_a16_moe = gemm_type != GemmType.DENSE and layer_config.a_dtype.num_bits == 16
        if not (is_nvfp4 and is_a16_moe and layer_config.num_experts) or shape_m <= 2:
            return None
        tokens_per_expert = shape_m / layer_config.num_experts
        tile = next((entry for entry in cls._NVFP4_MOE_TILES if tokens_per_expert <= entry[0]), None)
        if tile is None or tile[1] is None:
            return None
        _, block_shape, warp_shape, num_stages, num_ctas_per_sm = tile
        fits_shape = layer_config.shape_n % block_shape[1] == 0 and layer_config.shape_k % block_shape[2] == 0
        smem_size = estimate_smem_size_layer(layer_config, block_shape, gemm_type, num_stages)
        if not fits_shape or smem_size * num_ctas_per_sm > cls.max_smem_size:
            return None
        return {
            "block_shape": block_shape,
            "warp_shape": warp_shape,
            "num_stages": num_stages,
            "num_ctas_per_sm": num_ctas_per_sm,
        }

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        config = super().get_config(
            layer_config,
            shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if use_f16_accum or use_batch_invariant:
            return config
        tile = cls._get_nvfp4_moe_tile(layer_config, shape_m, gemm_type)
        if tile is not None:
            config.update(tile)
            config["evict_weights_first"] = cls._evict_weights_first(config, layer_config, shape_m, gemm_type)
        return config


class Sm89Heuristics(Sm86Heuristics):
    sm_version: int = 89
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]

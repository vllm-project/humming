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
        if cls._should_drop_stream_k_stage(layer_config, config, shape_m, gemm_type):
            config = dict(config, num_stages=config["num_stages"] - 1)

        tokens_per_expert = shape_m / (layer_config.num_experts or 1)
        block_m = config["block_shape"][0]
        config["evict_weights_first"] = tokens_per_expert <= block_m
        if gemm_type != GemmType.DENSE:
            config["evict_weights_first"] = tokens_per_expert <= block_m * 0.5 <= 8
        return config

    @classmethod
    def _stream_k_longest_iters(cls, layer_config: LayerConfig, config: dict, shape_m: int, num_stages: int):
        """K iterations of the busiest CTA, as Scheduler (scheduler.cuh) splits a dense Stream-K GEMM on SM80.

        Returns the iterations and whether every Stream-K slice covers whole tiles (no partial-sum fixup).
        """
        block_m, block_n, block_k = config["block_shape"]
        num_ctas = config["num_sms"] * config["num_ctas_per_sm"]
        group_size = max(layer_config.input_scale_group_size, layer_config.weight_scale_group_size, 1)
        blocks_per_group = max(1, group_size // block_k)
        k_blocks = ceil_div(layer_config.shape_k, block_k)
        stream_k_blocks = ceil_div(k_blocks, blocks_per_group) * blocks_per_group
        mn_blocks = ceil_div(shape_m, block_m) * ceil_div(layer_config.shape_n, block_n)
        stream_k_mn_blocks = mn_blocks
        if mn_blocks > num_ctas:
            stream_k_mn_blocks = mn_blocks % num_ctas
            if stream_k_mn_blocks and stream_k_mn_blocks * 10 <= num_ctas:
                stream_k_mn_blocks += num_ctas
        data_parallel_iters = (mn_blocks - stream_k_mn_blocks) // num_ctas * stream_k_blocks
        # At most 10 slices per tile, as in Scheduler.
        min_slice_iters = ceil_div(stream_k_blocks - 1, 9)
        slice_iters = max(ceil_div(stream_k_mn_blocks * stream_k_blocks, num_ctas), min_slice_iters)
        align_iters = blocks_per_group // math.gcd(blocks_per_group, num_stages) * num_stages
        slice_iters = ceil_div(slice_iters, align_iters) * align_iters
        # Slices past the last Stream-K tile are empty: with no tail (stream_k_mn_blocks == 0) no CTA
        # runs a Stream-K slice, and the busiest CTA never runs more than the whole tail.
        slice_iters = min(slice_iters, stream_k_mn_blocks * stream_k_blocks)
        return data_parallel_iters + slice_iters, slice_iters % stream_k_blocks == 0

    @classmethod
    def _should_drop_stream_k_stage(
        cls, layer_config: LayerConfig, config: dict, shape_m: int, gemm_type: GemmType
    ) -> bool:
        """Use 4 stages instead of 5 when that balances an INT3 Stream-K grid better.

        Scheduler rounds each CTA's Stream-K slice up to a multiple of
        lcm(group_size / block_k, num_stages). With 5 stages and block_k 64
        that is 10 iterations, so e.g. a 34816 x 5120 layer gives its busiest
        CTA 60 iterations instead of 52. INT3 weights are bound by each CTA's
        work, not by DRAM bandwidth, so that imbalance costs time directly
        (INT4 and wider weights barely gain from rebalancing).
        """
        is_dense_w3a16 = gemm_type == GemmType.DENSE and layer_config.a_dtype.num_bits == 16
        is_dense_w3a16 &= layer_config.b_dtype.num_bits == 3
        is_five_stage_stream_k = config["use_stream_k"] and config["num_stages"] == 5
        if cls.sm_version != 80 or not is_dense_w3a16 or not is_five_stage_stream_k:
            return False
        iters_5, whole_tiles_5 = cls._stream_k_longest_iters(layer_config, config, shape_m, 5)
        iters_4, whole_tiles_4 = cls._stream_k_longest_iters(layer_config, config, shape_m, 4)
        # A slice of whole tiles also skips the cross-CTA reduction, worth a few more iterations.
        return iters_5 >= 1.04 * iters_4 or (whole_tiles_4 and not whole_tiles_5 and iters_5 >= 0.9 * iters_4)

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
    sm_version: int = 87


class Sm89Heuristics(Sm86Heuristics):
    sm_version: int = 89
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]

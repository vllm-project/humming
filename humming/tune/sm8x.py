from humming import dtypes
from humming.config import GemmType
from humming.device import current_device
from humming.tune.base import DeviceHeuristics
from humming.utils.math import round_up
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
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        return cls._refine_mid_m_tile(config, layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type)

    @classmethod
    def _refine_mid_m_tile(
        cls,
        config: dict,
        layer_config,
        shape_m: int,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType,
    ) -> dict:
        """Cut M-padding of dense W4/W8A16 GEMMs for 128 < M <= 1024.

        The generic heuristic picks the fewest M blocks, which leaves up to
        one 96/128-row block of padding right after M = 128 * k. The kernel
        is close to compute bound there, so padded rows cost real time. Switch
        to a 32/48/64-row tile (2 CTAs/SM, Stream-K) only when it cuts the
        padded rows by at least 10%. Validated on RTX 4090 for g128 int4/FP8.
        """
        group_size = layer_config.input_scale_group_size or layer_config.weight_scale_group_size
        applicable = gemm_type == GemmType.DENSE and layer_config.a_dtype.num_bits == 16
        applicable &= layer_config.b_dtype.num_bits in (4, 8) and group_size == 128
        applicable &= not use_f16_accum and not use_batch_invariant
        applicable &= 128 < shape_m <= 1024
        applicable &= layer_config.shape_n % 128 == 0 and layer_config.shape_k % 64 == 0
        applicable &= layer_config.shape_k > 1024
        # 8-bit weights re-read per M block: with a long K (e.g. down_proj, K >= 14336) the
        # 48/64-row tile with warp_n=32 loses up to 40% on RTX 4090
        applicable &= layer_config.b_dtype.num_bits == 4 or layer_config.shape_k <= 12288
        if not applicable:
            return config

        old_pad = round_up(shape_m, config["block_shape"][0])
        # on a tie prefer the larger tile
        # 8-bit weights stream twice the bytes per M block, so 32-row tiles lose there
        candidates = (32, 48, 64) if layer_config.b_dtype.num_bits == 4 else (48, 64)
        new_pad, neg_block_m = min((round_up(shape_m, m), -m) for m in candidates)
        block_m = -neg_block_m
        if new_pad * 10 > old_pad * 9:
            return config

        block_shape = (block_m, 128, 64)
        warp_shape = (block_m, 32, 64)
        num_stages = 2
        smem_size = estimate_smem_size_layer(
            layer_config,
            block_shape,
            GemmType.DENSE,
            num_stages,
            warp_shape=warp_shape,
            mma_accum_bits=32,
            output_chunk_rows=config.get("output_chunk_rows", 0),
        )
        if smem_size * 2 > cls.max_smem_size:
            return config

        config = dict(config)
        config.update(
            block_shape=block_shape,
            warp_shape=warp_shape,
            use_stream_k=True,
            num_sms=current_device.sm_count,
            num_stages=num_stages,
            num_ctas_per_sm=2,
        )
        return config

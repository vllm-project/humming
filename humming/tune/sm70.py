from humming import dtypes
from humming.config import GemmType
from humming.tune.base import estimate_smem_size_layer
from humming.tune.sm75 import Sm75Heuristics


class Sm70Heuristics(Sm75Heuristics):
    """Volta: Turing's heuristics with fp16 activations only and tiles sized for `mma.sync.m8n8k4`.

    Volta keeps eight fp32 accumulators per 16x8 output tile (the two K halves of `m8n8k4` stay apart until
    the epilogue, see `utils/ptx/volta.cuh`), twice Turing's four, so warp tiles are kept at 16x64 or smaller.
    """

    max_smem_size: int = 96 * 1024
    sm_version: int = 70
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16]
    b8_allowed_dtypes: list[dtypes.DataType] = []
    b4_allowed_dtypes: list[dtypes.DataType] = []

    @classmethod
    def get_base_config(
        cls,
        a_dtype,
        b_dtype,
        group_size,
        use_f16_accum,
        use_fused_e8m0_scale,
        gemm_type,
        shape_k,
    ):
        base_args = (a_dtype, b_dtype, group_size, use_f16_accum, use_fused_e8m0_scale, gemm_type, shape_k)
        config = dict(super().get_base_config(*base_args))
        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        if warp_m * warp_n > 16 * 64:
            # Turing's 64x64 warp tile would need 256 accumulator registers on Volta.
            config["block_shape"] = (block_m // 2, block_n // 2, block_k)
            config["warp_shape"] = (16, 64, warp_k)
        return config

    @classmethod
    def get_config(
        cls,
        layer_config,
        shape_m,
        use_f16_accum=False,
        use_batch_invariant=False,
        gemm_type=None,
        use_m_major_input_scale=False,
    ):
        gemm_type = GemmType.DENSE if gemm_type is None else gemm_type
        config = super().get_config(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
            use_m_major_input_scale,
        )
        is_dense_a16 = gemm_type == GemmType.DENSE and layer_config.a_dtype.num_bits == 16
        if not is_dense_a16:
            return config

        block_m, block_n, block_k = config["block_shape"]
        warp_m, warp_n, warp_k = config["warp_shape"]
        if shape_m > 8 and warp_n == 32 and block_n % 64 == 0:
            # The Volta prompt path needs 64-column warp tiles; the base rule halves them for small M.
            warp_n = 64
            config["warp_shape"] = (warp_m, warp_n, warp_k)

        if shape_m > 16:
            return config

        # Decode: few output tiles (small N) and a long K get shorter K slices, so pipeline start-up does not
        # dominate.
        has_few_tiles = layer_config.shape_n // block_n <= 16
        if has_few_tiles and block_k > 64 and layer_config.shape_k % 64 == 0:
            block_k = 64
            warp_k = min(warp_k, 64)
            config["block_shape"] = (block_m, block_n, block_k)
            config["warp_shape"] = (warp_m, warp_n, warp_k)

        # One CTA of eight warps per SM leaves Volta idle between instructions: as many CTAs per SM as shared
        # memory and a 128-register budget allow (16 warps per SM for both 128- and 256-thread CTAs).
        threads = (block_m // warp_m) * (block_n // warp_n) * (block_k // warp_k) * 32
        smem_size = estimate_smem_size_layer(
            layer_config, config["block_shape"], gemm_type, config["num_stages"]
        )
        ctas_by_smem = max(1, cls.max_smem_size // max(smem_size, 1))
        ctas_by_registers = max(1, 65536 // (threads * 128))
        config["num_ctas_per_sm"] = max(config["num_ctas_per_sm"], min(ctas_by_smem, ctas_by_registers, 8))
        return config

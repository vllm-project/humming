"""Tile / split-K heuristics for the ROCm gfx950 MFMA weight-only GEMM.

Tuned standalone on MI355X (256 CUs); see docs/rocm/log_mfma.md. The tile only depends on the
M bucket and the layer shape (N, K, dtype, bits), so one layer compiles at most six kernels. The split-K
factor depends on the runtime M and is computed by the launcher (csrc/launcher/mfma.h::mfma_num_splits);
``get_num_splits`` is the Python mirror used by tests.
"""

import dataclasses
import math

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune.base import DeviceHeuristics

MAX_SPLITS = 8
CHUNK_K = 128


@dataclasses.dataclass(frozen=True)
class MfmaTile:
    warps_m: int = 1
    warps_n: int = 4
    tiles_m: int = 1
    tiles_n: int = 1
    stages: int = 4
    lds_a: bool = False
    scale_b: bool = False

    @property
    def block_m(self) -> int:
        return self.warps_m * self.tiles_m * 16

    @property
    def block_n(self) -> int:
        return self.warps_n * self.tiles_n * 16

    def to_tuning_config(self) -> dict:
        return {"mma_type": "mfma", "max_splits": MAX_SPLITS, **dataclasses.asdict(self)}

    def to_moe_tuning_config(self) -> dict:
        # block_shape[0] is the M block the routing must be aligned to (vLLM reads it as
        # moe_block_size); the kernel itself only reads the MfmaTile fields.
        return self.to_tuning_config() | {"block_shape": [self.block_m, self.block_n, CHUNK_K]}

    @classmethod
    def from_tuning_config(cls, config: dict) -> "MfmaTile":
        return cls(
            **{field.name: config[field.name] for field in dataclasses.fields(cls) if field.name in config}
        )


# The wide (8 x 4 MFMA tiles per wave, 128 x 256 block) tile halves the LDS A reads per MFMA
# and wins once its grid has >= WIDE_TILE_MIN_BLOCKS_PER_CU blocks per CU; below that the
# 128 x 128 tile (more blocks, split-K) is faster. Only FP16 with <= 4-bit weights uses it:
# the BF16 dequant (17 vs 13 VALU per fragment) and the 5..8-bit B fragments (twice the
# registers) make the 128-accumulator tile spill (2.5-3x slower). See docs/rocm/log_perf.md.
WIDE_TILE_MIN_BLOCKS_PER_CU = 1.5
WIDE_TILE_MAX_BITS = 4


def wide_tile_allowed(is_bfloat16: bool, num_bits: int) -> bool:
    return not is_bfloat16 and num_bits <= WIDE_TILE_MAX_BITS


def wide_tile_min_m(shape_n: int, num_cus: int) -> int:
    """M above which the wide tile is used (exclusive bound, a multiple of 128)."""
    n_blocks = math.ceil(shape_n / 256)
    m_blocks = math.ceil(WIDE_TILE_MIN_BLOCKS_PER_CU * num_cus / n_blocks)
    return max(64, 128 * (m_blocks - 1))


# M in (32, 64]: the 64 x 128 tile, unless split-K leaves it K-starved (fewer than
# MIN_CHUNKS_PER_SPLIT chunks of 128 per split); then the 32 x 128 tile (twice the blocks, half
# the splits) is 5-17% faster (small N and K, e.g. 4096x4096, 5120x5120, 6144x4096).
MIN_CHUNKS_PER_SPLIT = 8
TILE_M64 = MfmaTile(tiles_m=4, tiles_n=2, stages=3, lds_a=True, scale_b=True)
TILE_M32 = MfmaTile(tiles_m=2, tiles_n=2, stages=3, lds_a=True, scale_b=True)
# M in (16, 32] with small N (see use_narrow_m32_tile): 32 x 64 blocks, 5-25% faster.
TILE_M32_NARROW = MfmaTile(tiles_m=2, tiles_n=1, stages=3, lds_a=True, scale_b=True)
# M in (8, 16]: 16-row blocks instead of half-empty 32-row ones, 3-12% faster on every shape.
TILE_M16 = MfmaTile(tiles_m=1, tiles_n=2, stages=3, lds_a=True, scale_b=True)


def use_narrow_m32_tile(shape_n: int, shape_k: int, num_cus: int) -> bool:
    """32 x 64 instead of 32 x 128 blocks for M in (16, 32].

    Always for N <= num_cus / 8 * 128 (4096 on 256 CUs). For N up to num_cus / 4 * 128 only when
    K <= N: 5120x5120 and 6144x4096 gain 7-14%, but deeper K (6144x8192, 5120x25600) is up to 5%
    slower with 8-bit weights.
    """
    blocks_n = shape_n // 128
    return blocks_n <= num_cus // 8 or (blocks_n < num_cus // 4 and shape_k <= shape_n)


def is_k_starved(tile: MfmaTile, shape_m: int, shape_n: int, shape_k: int, num_cus: int) -> bool:
    num_splits = get_num_splits(tile, shape_m, shape_n, shape_k, num_cus)
    return shape_k // CHUNK_K // num_splits < MIN_CHUNKS_PER_SPLIT


def select_tile(
    shape_m: int, shape_n: int, shape_k: int, num_cus: int, is_bfloat16: bool, num_bits: int
) -> MfmaTile:
    if shape_m <= 8:
        # Decode: no LDS, one 16-row MFMA tile; wider N per wave only when N gives enough blocks.
        tiles_n = 2 if shape_n // 128 >= num_cus // 2 else 1
        return MfmaTile(tiles_m=1, tiles_n=tiles_n, stages=4)
    if shape_m <= 16:
        return TILE_M16
    if shape_m <= 32:
        return TILE_M32_NARROW if use_narrow_m32_tile(shape_n, shape_k, num_cus) else TILE_M32
    if shape_m <= 64:
        # 64 rows: one block row for every M in the bucket, so the choice is M-independent.
        return TILE_M32 if is_k_starved(TILE_M64, 64, shape_n, shape_k, num_cus) else TILE_M64
    if wide_tile_allowed(is_bfloat16, num_bits) and shape_m > wide_tile_min_m(shape_n, num_cus):
        return MfmaTile(tiles_m=8, tiles_n=4, stages=2, lds_a=True, scale_b=True)
    return MfmaTile(tiles_m=8, tiles_n=2, stages=2, lds_a=True, scale_b=True)


# M intervals (min exclusive, max inclusive) matching select_tile's buckets; the last bucket is
# split at wide_tile_min_m when the wide tile is allowed (see get_m_intervals).
M_INTERVALS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, 1 << 30))


def get_m_intervals(shape_n: int, num_cus: int, is_bfloat16: bool, num_bits: int) -> tuple:
    if not wide_tile_allowed(is_bfloat16, num_bits):
        return M_INTERVALS
    wide_min_m = wide_tile_min_m(shape_n, num_cus)
    if wide_min_m <= 64:
        return M_INTERVALS
    return M_INTERVALS[:-1] + ((64, wide_min_m), (wide_min_m, 1 << 30))


# MoE: the M block is chosen from the routed rows per expert (valid_shape_m = tokens * top_k
# spread over the layer's experts), never from N, so that the w13 and w2 GEMMs of one layer pick
# the same block_m for one valid_shape_m: vLLM aligns the routing once (moe_align_block_size with
# the w13 config's block_shape[0]) and reuses it for w2.
# Thresholds on valid_shape_m (tokens * top_k) itself, then on rows per expert
# (valid_shape_m / num_experts). Tuned on Mixtral / Qwen3-30B-A3B / DeepSeek-V3 shapes, W4 g128,
# see docs/rocm/log_moe.md.
MOE_DIRECT_A_MAX_ROWS = 16
MOE_ROWS_PER_EXPERT_BUCKETS = (8, 16, 256)


def select_moe_tile(valid_shape_m: int, num_experts: int) -> MfmaTile:
    num_experts = max(num_experts, 1)
    if valid_shape_m <= MOE_DIRECT_A_MAX_ROWS:
        # A handful of routed rows, ~1 per active expert: A straight from global memory (an LDS
        # copy of a 16-row block that holds one valid row is pure overhead).
        return MfmaTile(tiles_m=1, tiles_n=1, stages=2)
    rows_per_expert = math.ceil(valid_shape_m / num_experts)
    if rows_per_expert <= 8:
        # Several rows per block: stage A once in LDS for the 4 waves.
        return MfmaTile(tiles_m=1, tiles_n=1, stages=3, lds_a=True)
    if rows_per_expert <= 16:
        return MfmaTile(tiles_m=2, tiles_n=1, stages=3, lds_a=True)
    if rows_per_expert <= 256:
        return MfmaTile(tiles_m=4, tiles_n=2, stages=2, lds_a=True)
    return MfmaTile(tiles_m=8, tiles_n=2, stages=2, lds_a=True, scale_b=True)


def moe_m_intervals(num_experts: int) -> list[tuple[int, int]]:
    """valid_shape_m intervals (min exclusive, max inclusive), each inside one select_moe_tile case."""
    num_experts = max(num_experts, 1)
    thresholds = {MOE_DIRECT_A_MAX_ROWS} | {rows * num_experts for rows in MOE_ROWS_PER_EXPERT_BUCKETS}
    bounds = [0] + sorted(thresholds) + [1 << 30]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def get_num_splits(
    tile: MfmaTile, shape_m: int, shape_n: int, shape_k: int, num_cus: int, max_splits: int = MAX_SPLITS
) -> int:
    """Split K until about one block per CU (each split keeps >= 4 chunks of 128)."""
    num_blocks = math.ceil(shape_n / tile.block_n) * math.ceil(shape_m / tile.block_m)
    num_chunks_k = shape_k // CHUNK_K
    num_splits = 1
    while (
        num_blocks * num_splits < num_cus
        and num_splits < max_splits
        and num_chunks_k // (num_splits * 2) >= 4
    ):
        num_splits *= 2
    return num_splits


class Gfx950Heuristics(DeviceHeuristics):
    @classmethod
    def num_cus(cls) -> int:
        from humming.device import current_device

        return current_device.sm_count

    @classmethod
    def should_use_pdl_for_input(cls, layer_config: LayerConfig, shape_m: int) -> bool:
        return False

    @classmethod
    def _check(cls, layer_config: LayerConfig, gemm_type) -> GemmType:
        from humming.kernel.humming_mfma import check_mfma_gemm_type, check_mfma_layer_config

        check_mfma_layer_config(layer_config)
        return check_mfma_gemm_type(layer_config, gemm_type)

    @classmethod
    def get_config(cls, layer_config: LayerConfig, shape_m: int, gemm_type=GemmType.DENSE, **kwargs) -> dict:
        if cls._check(layer_config, gemm_type) != GemmType.DENSE:
            return select_moe_tile(shape_m, layer_config.num_experts).to_moe_tuning_config()
        if layer_config.a_dtype.num_bits < 16:
            from humming.kernel.humming_mfma_f8 import plan_mfma_f8, select_f8_tile

            plan = plan_mfma_f8(layer_config)
            return select_f8_tile(shape_m, layer_config.shape_n, cls.num_cus(), plan).to_tuning_config()
        is_bfloat16 = layer_config.a_dtype == dtypes.bfloat16
        num_bits = layer_config.b_dtype.num_bits
        shape_n, shape_k = layer_config.shape_n, layer_config.shape_k
        return select_tile(shape_m, shape_n, shape_k, cls.num_cus(), is_bfloat16, num_bits).to_tuning_config()

    @classmethod
    def get_configs(cls, layer_config: LayerConfig, gemm_type=GemmType.DENSE, **kwargs) -> list:
        if cls._check(layer_config, gemm_type) != GemmType.DENSE:
            return [
                [low, high, select_moe_tile(high, layer_config.num_experts).to_moe_tuning_config()]
                for low, high in moe_m_intervals(layer_config.num_experts)
            ]
        if layer_config.a_dtype.num_bits < 16:
            from humming.kernel.humming_mfma_f8 import F8_M_INTERVALS as intervals
        else:
            is_bfloat16 = layer_config.a_dtype == dtypes.bfloat16
            intervals = get_m_intervals(
                layer_config.shape_n, cls.num_cus(), is_bfloat16, layer_config.b_dtype.num_bits
            )
        return [[low, high, cls.get_config(layer_config, high, gemm_type)] for low, high in intervals]

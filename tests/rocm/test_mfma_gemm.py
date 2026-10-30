"""Correctness of the gfx950 MFMA weight-only GEMM against an FP32 torch reference."""

import dataclasses

import pytest
import torch

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.tune import gfx950
from tests.rocm.mfma_gemm import MfmaTile, mfma_gemm, select_tile
from tests.rocm.quant_utils import TOLERANCE, error_stats, make_quantized_linear


def run_case(
    shape_m, shape_n, shape_k, num_bits, group_size, has_zero_point, dtype, tile=None, num_splits=None
):
    linear = make_quantized_linear(shape_n, shape_k, num_bits, group_size, has_zero_point, dtype)
    inputs = torch.randn(shape_m, shape_k, dtype=dtype, device="cuda")
    outputs = mfma_gemm(
        inputs,
        linear.mfma_weight,
        linear.mfma_scale,
        linear.mfma_zero_point,
        num_bits,
        group_size,
        tile=tile,
        num_splits=num_splits,
    )
    reference = inputs.float() @ linear.dequantized().T
    return error_stats(outputs, reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_bits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [0, 32, 64, 128])
@pytest.mark.parametrize("has_zero_point", [False, True])
@pytest.mark.parametrize("shape_m", [1, 7, 16, 32, 64, 255])
def test_small_shapes(dtype, num_bits, group_size, has_zero_point, shape_m):
    if has_zero_point and group_size == 0:
        pytest.skip("zero point is tested group-wise")
    _, rel = run_case(shape_m, 256, 1024, num_bits, group_size, has_zero_point, dtype)
    assert rel < TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tile",
    [
        MfmaTile(1, 4, 1, 1, 4),
        MfmaTile(1, 4, 2, 2, 2),
        MfmaTile(2, 2, 4, 2, 2),
        MfmaTile(4, 1, 1, 4, 3),
        MfmaTile(2, 2, 4, 4, 2, lds_a=True),
        MfmaTile(1, 4, 2, 2, 3, lds_a=True),
        MfmaTile(1, 4, 8, 2, 2, lds_a=True, scale_b=True),
        MfmaTile(1, 4, 8, 4, 2, lds_a=True, scale_b=True),
        MfmaTile(1, 4, 2, 2, 3, lds_a=True, scale_b=True),
        MfmaTile(1, 4, 4, 2, 3, lds_a=True, scale_b=True),
        MfmaTile(1, 4, 1, 2, 3, lds_a=True, scale_b=True),
        MfmaTile(1, 4, 2, 1, 3, lds_a=True, scale_b=True),
    ],
)
@pytest.mark.parametrize("num_splits", [1, 2, 3, 8])
def test_tiles_and_splits(dtype, tile, num_splits):
    _, rel = run_case(100, 512, 2048, 4, 128, True, dtype, tile=tile, num_splits=num_splits)
    assert rel < TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_bits", [2, 3, 4, 8])
@pytest.mark.parametrize("group_size", [0, 32, 128])
@pytest.mark.parametrize("shape_m", [1, 77, 300])
@pytest.mark.parametrize("scale_b", [False, True])
def test_lds_a_tile(dtype, num_bits, group_size, shape_m, scale_b):
    tile = (
        MfmaTile(1, 4, 8, 2, 2, lds_a=True, scale_b=scale_b)
        if scale_b
        else MfmaTile(2, 2, 4, 4, 2, lds_a=True)
    )
    _, rel = run_case(shape_m, 384, 1024, num_bits, group_size, group_size > 0, dtype, tile=tile)
    assert rel < TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_bits", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("group_size", [0, 128])
@pytest.mark.parametrize("has_zero_point", [False, True])
@pytest.mark.parametrize("shape_m", [77, 300])
def test_wide_tile(dtype, num_bits, group_size, has_zero_point, shape_m):
    """The 128 x 256 tile picked for large-M FP16 (tune/gfx950.py), also checked with BF16."""
    if has_zero_point and group_size == 0:
        pytest.skip("zero point is tested group-wise")
    tile = MfmaTile(1, 4, 8, 4, 2, lds_a=True, scale_b=True)
    _, rel = run_case(shape_m, 768, 1024, num_bits, group_size, has_zero_point, dtype, tile=tile)
    assert rel < TOLERANCE[dtype]


@pytest.mark.parametrize("shape_k", [4096, 14336])
@pytest.mark.parametrize("num_bits", [1, 4, 5, 8])
@pytest.mark.parametrize("is_bfloat16", [False, True])
@pytest.mark.parametrize("shape_n", [1024, 4096, 8192, 28672, 57344, 131072])
def test_tile_buckets(is_bfloat16, shape_n, num_bits, shape_k):
    """Each M interval of the tuner maps to one tile, and the test mirror agrees with it."""
    num_cus = 256
    intervals = gfx950.get_m_intervals(shape_n, num_cus, is_bfloat16, num_bits)
    assert intervals[0][0] == 0 and intervals[-1][1] == 1 << 30
    for (_, high), (low, _) in zip(intervals[:-1], intervals[1:], strict=True):
        assert high == low
    for low, high in intervals:
        probes = {low + 1, min(high, low + 129), min(high, 1 << 16)}
        tiles = {gfx950.select_tile(m, shape_n, shape_k, num_cus, is_bfloat16, num_bits) for m in probes}
        assert len(tiles) == 1, (low, high, tiles)
        for m in probes:
            mirror, _ = select_tile(m, shape_n, shape_k, num_bits, num_cus, is_bfloat16)
            assert dataclasses.asdict(mirror) == dataclasses.asdict(next(iter(tiles)))
    wide = gfx950.select_tile(1 << 16, shape_n, shape_k, num_cus, is_bfloat16, num_bits)
    assert wide.tiles_n == (4 if not is_bfloat16 and num_bits <= 4 else 2)


@pytest.mark.parametrize(
    "shape_n,shape_k,tiles_m",
    [(4096, 4096, 2), (6144, 4096, 2), (5120, 5120, 2), (4096, 14336, 4), (8192, 8192, 4), (28672, 4096, 4)],
)
def test_m64_tile_k_starved(shape_n, shape_k, tiles_m):
    """M in (32, 64] falls back to the 32x128 tile when the 64x128 grid has < 8 K chunks per split."""
    for shape_m in (33, 48, 64):
        tile = gfx950.select_tile(shape_m, shape_n, shape_k, 256, False, 4)
        assert (tile.tiles_m, tile.tiles_n) == (tiles_m, 2)
        mirror, _ = select_tile(shape_m, shape_n, shape_k, 4, 256)
        assert dataclasses.asdict(mirror) == dataclasses.asdict(tile)


@pytest.mark.parametrize(
    "shape_n,shape_k,narrow",
    [
        (4096, 4096, True),
        (4096, 28672, True),
        (5120, 5120, True),
        (6144, 4096, True),
        (6144, 8192, False),
        (5120, 25600, False),
        (8192, 4096, False),
        (57344, 8192, False),
    ],
)
def test_small_m_tiles(shape_n, shape_k, narrow):
    """M <= 16 uses 16-row blocks; M in (16, 32] uses 32 x 64 blocks for small N (and K <= N above 4096)."""
    for shape_m in (9, 12, 16):
        tile = gfx950.select_tile(shape_m, shape_n, shape_k, 256, False, 4)
        assert (tile.tiles_m, tile.tiles_n, tile.lds_a) == (1, 2, True)
    for shape_m in (17, 24, 32):
        tile = gfx950.select_tile(shape_m, shape_n, shape_k, 256, False, 4)
        assert (tile.tiles_m, tile.tiles_n) == (2, 1 if narrow else 2)
        mirror, _ = select_tile(shape_m, shape_n, shape_k, 4, 256)
        assert dataclasses.asdict(mirror) == dataclasses.asdict(tile)


@pytest.mark.parametrize("num_bits", [2, 4, 5, 8])
@pytest.mark.parametrize("a_dtype", [dtypes.float16, dtypes.bfloat16])
def test_heuristics_wide_tile_by_dtype(monkeypatch, a_dtype, num_bits):
    """The layer tuner only picks the 128x256 tile for FP16 activations with <= 4-bit weights."""
    monkeypatch.setattr(gfx950.Gfx950Heuristics, "num_cus", classmethod(lambda cls: 256))
    layer_config = LayerConfig(
        shape_n=28672,
        shape_k=8192,
        a_dtype=a_dtype,
        b_dtype=dtypes.DataType.from_str(f"uint{num_bits}"),
        c_dtype=a_dtype,
        bs_dtype=a_dtype,
        weight_scale_group_size=128,
    )
    configs = gfx950.Gfx950Heuristics.get_configs(layer_config, gemm_type=GemmType.DENSE)
    tiles_n = {config["tiles_n"] for low, _, config in configs if low >= 64}
    expect_wide = a_dtype == dtypes.float16 and num_bits <= 4
    assert tiles_n == ({2, 4} if expect_wide else {2})
    config = gfx950.Gfx950Heuristics.get_config(layer_config, 4096, gemm_type=GemmType.DENSE)
    assert config["tiles_n"] == (4 if expect_wide else 2)


def test_split_k_repeated_launches():
    """Split-K counters must reset so back-to-back launches stay correct."""
    dtype = torch.float16
    linear = make_quantized_linear(1024, 4096, 4, 128, False, dtype)
    inputs = torch.randn(1, 4096, dtype=dtype, device="cuda")
    reference = inputs.float() @ linear.dequantized().T
    for _ in range(20):
        outputs = mfma_gemm(inputs, linear.mfma_weight, linear.mfma_scale, None, 4, 128, num_splits=8)
        assert error_stats(outputs, reference)[1] < TOLERANCE[dtype]

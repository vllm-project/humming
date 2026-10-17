import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.sm90_h20 import Sm90H20Heuristics


@pytest.fixture(autouse=True)
def _h20_num_sms(monkeypatch):
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 78))


def _layer(shape_n: int, shape_k: int, num_experts: int = 0) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.int4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=0,
        sm_version=90,
    )


def _mxfp4_layer(shape_n: int, shape_k: int, a_dtype=dtypes.bfloat16) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=96,
        a_dtype=a_dtype,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=32,
        sm_version=90,
    )


def test_short_k_dense_uses_small_m_tiles_without_tma():
    config = Sm90H20Heuristics.get_config(_layer(512, 512), 64)

    assert config["block_shape"][0] == 8
    assert not config.get("use_tma", False)


@pytest.mark.parametrize(
    ("shape_k", "use_tma"),
    [(1024, False), (1536, True)],
)
def test_dense_tma_requires_enough_pipeline_turns(shape_k, use_tma):
    config = Sm90H20Heuristics.get_config(_layer(8192, shape_k), 256)

    assert config.get("use_tma", False) is use_tma


def test_moderate_stream_k_keeps_four_stages():
    config = Sm90H20Heuristics.get_config(_layer(2048, 2048), 8)

    assert config["use_stream_k"]
    assert config["num_stages"] == 4


def test_long_k_balances_m_reuse_and_stream_k_grid():
    config = Sm90H20Heuristics.get_config(_layer(512, 8192), 32)

    assert config["block_shape"][0] == 16
    assert config["num_stages"] == 4
    assert config["num_sms"] >= 64


@pytest.mark.parametrize("shape_n", [512, 1024])
@pytest.mark.parametrize("gemm_type", [GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS])
def test_sparse_long_k_moe_uses_mma_without_extra_row_padding(shape_n, gemm_type):
    layer = _layer(shape_n, 7168, num_experts=256)
    sparse = Sm90H20Heuristics.get_config(layer, 8, gemm_type=gemm_type)
    populated = Sm90H20Heuristics.get_config(layer, 1024, gemm_type=gemm_type)

    assert sparse["mma_type"] == "mma"
    assert sparse["block_shape"][0] == sparse["warp_shape"][0] == 8
    assert populated.get("mma_type", "wgmma") == "wgmma"


@pytest.mark.parametrize(
    ("shape_m", "block_shape", "warp_shape", "num_ctas", "use_stream_k"),
    [
        (3072, (40, 256, 64), (40, 64, 64), 2, True),
        (3840, (48, 128, 64), (48, 32, 64), 3, False),
        (12288, (48, 128, 64), (48, 32, 64), 3, False),
        (24576, (48, 128, 64), (48, 32, 64), 3, False),
    ],
)
@pytest.mark.parametrize("a_dtype", [dtypes.float16, dtypes.bfloat16])
def test_mxfp4_grouped_contiguous_selects_residency_for_expert_tiles(
    shape_m, block_shape, warp_shape, num_ctas, use_stream_k, a_dtype
):
    config = Sm90H20Heuristics.get_config(
        _mxfp4_layer(4608, 5120, a_dtype),
        shape_m,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )

    assert config["block_shape"] == block_shape
    assert config["warp_shape"] == warp_shape
    assert config["num_ctas_per_sm"] == num_ctas
    assert config["use_stream_k"] is use_stream_k


@pytest.mark.parametrize("a_dtype", [dtypes.float16, dtypes.bfloat16, dtypes.float8e4m3, dtypes.int8])
def test_small_mxfp4_grouped_contiguous_keeps_small_expert_tiles(a_dtype):
    config = Sm90H20Heuristics.get_config(
        _mxfp4_layer(4608, 5120, a_dtype),
        2048,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
    )

    assert config["block_shape"][0] == 32

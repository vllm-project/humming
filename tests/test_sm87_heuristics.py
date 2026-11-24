import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.sm8x import Sm87Heuristics
from humming.utils.smem import estimate_smem_size_layer


@pytest.fixture(autouse=True)
def _mock_jetson_agx_orin(monkeypatch):
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 16))
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 87))


def _shared_sm8x_config(layer: LayerConfig, **kwargs) -> dict:
    """The config the shared SM8x rules give SM87, i.e. its behavior before the NVFP4 MoE tiles."""
    return super(Sm87Heuristics, Sm87Heuristics).get_config(layer, **kwargs)


def _nvfp4_moe_layer(shape_n: int, shape_k: int, num_experts: int = 128) -> LayerConfig:
    return LayerConfig(
        sm_version=87,
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e4m3,
        weight_scale_group_size=16,
    )


@pytest.mark.parametrize("shape_n,shape_k", [(2304, 2304), (2304, 1152)])
@pytest.mark.parametrize(
    "shape_m,block_shape,num_ctas_per_sm",
    [
        (8, (8, 128, 128), 2),
        (144, (8, 128, 128), 2),
        (1024, (16, 256, 64), 2),
        (1840, (16, 256, 64), 2),
        (2048, (32, 256, 64), 2),
        (3808, (32, 256, 64), 2),
        (4096, (64, 256, 64), 1),
        (8192, (64, 256, 64), 1),
    ],
)
def test_nvfp4_moe_uses_orin_tiles(shape_n, shape_k, shape_m, block_shape, num_ctas_per_sm):
    layer = _nvfp4_moe_layer(shape_n, shape_k)
    config = Sm87Heuristics.get_config(layer, shape_m=shape_m, gemm_type=GemmType.INDEXED)
    assert config["block_shape"] == block_shape
    assert config["num_stages"] == 4
    assert config["num_ctas_per_sm"] == num_ctas_per_sm
    smem_size = estimate_smem_size_layer(layer, config["block_shape"], GemmType.INDEXED, config["num_stages"])
    assert smem_size * config["num_ctas_per_sm"] <= Sm87Heuristics.max_smem_size
    # The evict-first hint must describe the Orin tile, not the SM80 tile it replaced.
    tokens_per_expert = shape_m / 128
    assert config["evict_weights_first"] == (tokens_per_expert <= block_shape[0] * 0.5 <= 8)


@pytest.mark.parametrize("shape_m", [1, 2, 160, 512, 912, 8320, 16384])
def test_nvfp4_moe_outside_measured_ranges_keeps_shared_tiles(shape_m):
    layer = _nvfp4_moe_layer(2304, 2304)
    kwargs = dict(shape_m=shape_m, gemm_type=GemmType.INDEXED)
    assert Sm87Heuristics.get_config(layer, **kwargs) == _shared_sm8x_config(layer, **kwargs)


@pytest.mark.parametrize(
    "layer,gemm_type,kwargs",
    [
        (_nvfp4_moe_layer(2304, 2304, num_experts=0), GemmType.DENSE, {}),
        (_nvfp4_moe_layer(2304, 2304), GemmType.INDEXED, {"use_batch_invariant": True}),
        (_nvfp4_moe_layer(2304, 2304), GemmType.INDEXED, {"use_f16_accum": True}),
        (
            LayerConfig(
                sm_version=87,
                shape_n=2304,
                shape_k=2304,
                num_experts=128,
                a_dtype=dtypes.bfloat16,
                b_dtype=dtypes.uint4,
                c_dtype=dtypes.bfloat16,
                bs_dtype=dtypes.bfloat16,
                weight_scale_group_size=128,
            ),
            GemmType.INDEXED,
            {},
        ),
    ],
)
def test_other_layers_keep_shared_tiles(layer, gemm_type, kwargs):
    for shape_m in (8, 1024, 4096):
        new = Sm87Heuristics.get_config(layer, shape_m=shape_m, gemm_type=gemm_type, **kwargs)
        shared = _shared_sm8x_config(layer, shape_m=shape_m, gemm_type=gemm_type, **kwargs)
        assert new == shared

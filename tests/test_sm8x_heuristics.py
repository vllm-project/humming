import dataclasses

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.sm8x import Sm80Heuristics, Sm86Heuristics, Sm87Heuristics, Sm89Heuristics
from humming.tune.sm100 import Sm100MmaHeuristics
from humming.utils.smem import estimate_smem_size_layer


@pytest.fixture(autouse=True)
def _mock_rtx3080_device(monkeypatch):
    tensorcore_tops = {
        "float16": 61.1,
        "bfloat16": 61.1,
        "int8": 244.4,
        "int4": 488.8,
        "float8e4m3": 122.2,
        "float8e5m2": 122.2,
    }
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 68))
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 86))
    monkeypatch.setattr(DeviceInfo, "memory_bandwidth_gbps", property(lambda self: 760.0))
    monkeypatch.setattr(DeviceInfo, "tensorcore_tops", property(lambda self: tensorcore_tops))


@pytest.mark.parametrize(
    "heuristics_cls,a_dtype,b_dtype",
    [
        (heuristics_cls, a_dtype, b_dtype)
        for heuristics_cls in (Sm86Heuristics, Sm89Heuristics)
        for a_dtype in (dtypes.float16, dtypes.bfloat16)
        for b_dtype in (dtypes.int8, dtypes.float8e4m3, dtypes.int4)
    ]
    + [
        (Sm89Heuristics, dtypes.float8e4m3, dtypes.float8e5m2),
        (Sm89Heuristics, dtypes.float8e5m2, dtypes.float8e4m3),
        (Sm89Heuristics, dtypes.float8e5m2, dtypes.uint5),
    ],
)
@pytest.mark.parametrize("weight_scale_group_size", [0, 32, 128])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("shape_m", [1, 1024, 16384])
def test_config_fits_in_smem(heuristics_cls, a_dtype, b_dtype, weight_scale_group_size, has_bias, shape_m):
    param_dtype = a_dtype if a_dtype.num_bits == 16 else dtypes.bfloat16
    layer_config = LayerConfig(
        sm_version=heuristics_cls.sm_version,
        shape_n=4096,
        shape_k=4096,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=param_dtype,
        bs_dtype=param_dtype,
        weight_scale_group_size=weight_scale_group_size,
        has_bias=has_bias,
    )

    config = heuristics_cls.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE)

    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        GemmType.DENSE,
        config["num_stages"],
    )
    assert smem_size * config["num_ctas_per_sm"] <= heuristics_cls.max_smem_size
    if a_dtype.num_bits == b_dtype.num_bits == 8:
        same_type_layer = dataclasses.replace(layer_config, b_dtype=a_dtype)
        same_type_config = heuristics_cls.get_config(same_type_layer, shape_m, gemm_type=GemmType.DENSE)
        assert config == same_type_config


def _block_fp8_moe_layer(sm_version: int) -> LayerConfig:
    # Routed experts of one EP rank: 64 local experts, hidden 2560, intermediate 640, block FP8 weights.
    return LayerConfig(
        sm_version=sm_version,
        shape_n=1280,
        shape_k=2560,
        num_experts=64,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.float8e4m3,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float32,
        weight_scale_group_size=128,
        weight_scale_group_size_n=128,
    )


@pytest.mark.parametrize("heuristics_cls", [Sm80Heuristics, Sm87Heuristics])
def test_sm80_evicts_weights_first_for_single_m_tile(heuristics_cls):
    layer_config = _block_fp8_moe_layer(heuristics_cls.sm_version)

    small = heuristics_cls.get_config(layer_config, shape_m=8, gemm_type=GemmType.INDEXED)
    large = heuristics_cls.get_config(layer_config, shape_m=65536, gemm_type=GemmType.INDEXED)

    assert small["evict_weights_first"]
    assert not large["evict_weights_first"]


def test_sm100_mma_keeps_default_weight_eviction():
    layer_config = _block_fp8_moe_layer(Sm100MmaHeuristics.sm_version)

    config = Sm100MmaHeuristics.get_config(layer_config, shape_m=8, gemm_type=GemmType.INDEXED)

    assert not config.get("evict_weights_first", False)

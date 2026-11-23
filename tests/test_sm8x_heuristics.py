import dataclasses

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.sm8x import Sm86Heuristics, Sm89Heuristics
from humming.utils.math import round_up
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


def _mid_m_layer_config(b_dtype, shape_n=4096, shape_k=4096):
    return LayerConfig(
        sm_version=89,
        shape_n=shape_n,
        shape_k=shape_k,
        a_dtype=dtypes.bfloat16,
        b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
    )


def _sm89_and_base_configs(layer_config, shape_m):
    config = Sm89Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE)
    base_config = Sm86Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE)
    return config, base_config


@pytest.mark.parametrize("b_dtype", [dtypes.int4, dtypes.float8e4m3])
@pytest.mark.parametrize("shape_m", [129, 385])
def test_sm89_mid_m_uses_smaller_block_m(b_dtype, shape_m):
    layer_config = _mid_m_layer_config(b_dtype)
    config, base_config = _sm89_and_base_configs(layer_config, shape_m)

    assert config["block_shape"][0] in (32, 48, 64)
    assert config["block_shape"][0] < base_config["block_shape"][0]
    assert config["use_stream_k"]
    assert config["num_ctas_per_sm"] == 2
    padded_m = round_up(shape_m, config["block_shape"][0])
    base_padded_m = round_up(shape_m, base_config["block_shape"][0])
    assert padded_m * 10 <= base_padded_m * 9


@pytest.mark.parametrize("b_dtype", [dtypes.int4, dtypes.float8e4m3])
def test_sm89_mid_m_keeps_config_at_m128(b_dtype):
    layer_config = _mid_m_layer_config(b_dtype)
    config, base_config = _sm89_and_base_configs(layer_config, shape_m=128)
    assert config == base_config


@pytest.mark.parametrize(
    "b_dtype,shape_n,shape_k",
    [
        (dtypes.float8e4m3, 8192, 14336),
        (dtypes.int4, 4160, 4096),
        (dtypes.float8e4m3, 4160, 4096),
        (dtypes.int4, 4096, 1024),
        (dtypes.float8e4m3, 4096, 1024),
    ],
)
@pytest.mark.parametrize("shape_m", [129, 385])
def test_sm89_mid_m_not_applicable(b_dtype, shape_n, shape_k, shape_m):
    layer_config = _mid_m_layer_config(b_dtype, shape_n=shape_n, shape_k=shape_k)
    config, base_config = _sm89_and_base_configs(layer_config, shape_m)
    assert config == base_config


def test_sm89_mid_m_int4_long_k_still_applies():
    layer_config = _mid_m_layer_config(dtypes.int4, shape_n=8192, shape_k=14336)
    config, base_config = _sm89_and_base_configs(layer_config, shape_m=129)
    assert config["block_shape"][0] < base_config["block_shape"][0]

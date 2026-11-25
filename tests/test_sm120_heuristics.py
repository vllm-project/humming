import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, TuningConfig
from humming.config.enum import InputQuantizationMode
from humming.config.mma import get_register_budget_error
from humming.device import DeviceInfo
from humming.tune.sm120 import Sm120Heuristics


@pytest.fixture(autouse=True)
def _mock_rtx5090_device(monkeypatch):
    tensorcore_tops = {
        "float16": 419.0,
        "bfloat16": 419.0,
        "float8e4m3": 838.0,
        "float8e5m2": 838.0,
        "int8": 838.0,
        "int4": 1676.0,
    }
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 170))
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 120))
    monkeypatch.setattr(DeviceInfo, "max_registers_per_sm", property(lambda self: 65536))
    monkeypatch.setattr(DeviceInfo, "max_smem_size", property(lambda self: 101376))
    monkeypatch.setattr(DeviceInfo, "memory_bandwidth_gbps", property(lambda self: 1792.0))
    monkeypatch.setattr(DeviceInfo, "tensorcore_tops", property(lambda self: tensorcore_tops))


@pytest.mark.parametrize("shape_n", [1536, 2048])
@pytest.mark.parametrize("use_batch_invariant", [False, True])
def test_dense_mxfp8_fitted_m_tile_fits_register_budget(shape_n, use_batch_invariant):
    layer_config = LayerConfig(
        sm_version=120,
        shape_n=shape_n,
        shape_k=1024,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float8e4m3,
        c_dtype=dtypes.bfloat16,
        as_dtype=dtypes.float8e8m0,
        bs_dtype=dtypes.float8e8m0,
        input_quant_mode=InputQuantizationMode.DynamicGroup,
        input_scale_group_size=32,
        weight_scale_group_size=32,
    )
    # Tile choices change at multiples of 16 rows, so sample every bucket.
    # 1233..1248 rows (N=2048) used to fit a single 208-row M warp.
    for shape_m in range(8, 4097, 16):
        config = Sm120Heuristics.get_config(
            layer_config,
            shape_m,
            use_batch_invariant=use_batch_invariant,
            gemm_type=GemmType.DENSE,
        )
        tuning_config = TuningConfig(
            block_shape=config["block_shape"],
            warp_shape=config["warp_shape"],
            num_ctas_per_sm=config.get("num_ctas_per_sm", 1),
            use_warp_spec=config.get("use_warp_spec", False),
        )
        assert get_register_budget_error(layer_config, tuning_config) is None, (shape_m, config)

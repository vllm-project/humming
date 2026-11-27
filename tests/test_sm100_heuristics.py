import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.config.enum import WeightScale2Type, WeightScaleType
from humming.device import DeviceInfo
from humming.tune.candidate import DeviceProfile, ScheduleCandidate, TuningProblem, analyze_candidate
from humming.tune.sm100 import Sm100Heuristics, Sm100MmaHeuristics
from humming.utils.smem import estimate_smem_size_layer


@pytest.fixture(autouse=True)
def _mock_b200_device(monkeypatch):
    tensorcore_tops = {"float16": 2250.0, "bfloat16": 2250.0, "int8": 4500.0, "float8e4m3": 4500.0}
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 148))
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 100))
    monkeypatch.setattr(DeviceInfo, "memory_bandwidth_gbps", property(lambda self: 8000.0))
    monkeypatch.setattr(DeviceInfo, "tensorcore_tops", property(lambda self: tensorcore_tops))


def _w4a16_layer_config(shape_n, shape_k, b_dtype=dtypes.uint4, sm_version=100):
    return LayerConfig(
        sm_version=sm_version,
        shape_n=shape_n,
        shape_k=shape_k,
        a_dtype=dtypes.bfloat16,
        b_dtype=b_dtype,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
    )


# Weight layouts besides int4 g128, as vLLM passes them to Humming for FP8 and NVFP4 checkpoints.
A16_WEIGHTS = {
    "fp8": {"b_dtype": dtypes.float8e4m3, "weight_scale_type": WeightScaleType.CHANNEL},
    "fp8_block": {
        "b_dtype": dtypes.float8e4m3,
        "bs_dtype": dtypes.float32,
        "weight_scale_group_size": 128,
        "weight_scale_group_size_n": 128,
        "weight_scale_type": WeightScaleType.BLOCK,
    },
    "fp8_e5m2": {"b_dtype": dtypes.float8e5m2, "weight_scale_type": WeightScaleType.CHANNEL},
    "nvfp4": {
        "b_dtype": dtypes.float4e2m1,
        "bs_dtype": dtypes.float8e4m3,
        "weight_scale_group_size": 16,
        "weight_scale_type": WeightScaleType.GROUP,
        "weight_scale_2_type": WeightScale2Type.TENSOR,
    },
    "mxfp4": {
        "b_dtype": dtypes.float4e2m1,
        "bs_dtype": dtypes.float8e8m0,
        "weight_scale_group_size": 32,
        "weight_scale_type": WeightScaleType.GROUP,
    },
}


def _a16_layer_config(shape_n, shape_k, weights):
    if weights == "int4":
        return _w4a16_layer_config(shape_n, shape_k)
    if weights == "int8":
        return _w4a16_layer_config(shape_n, shape_k, dtypes.int8)
    return LayerConfig(
        sm_version=100,
        shape_n=shape_n,
        shape_k=shape_k,
        a_dtype=dtypes.bfloat16,
        c_dtype=dtypes.bfloat16,
        **({"bs_dtype": dtypes.bfloat16} | A16_WEIGHTS[weights]),
    )


def _get_config(monkeypatch, layer_config, shape_m, **kwargs):
    # Under pytest the heuristic prefers UMMA whenever the layer supports it; select as in production.
    monkeypatch.delenv("HUMMING_TEST_TUNING_SOURCE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    return Sm100Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE, **kwargs)


def _get_config_without_decode_tile(monkeypatch, layer_config, shape_m, **kwargs):
    with monkeypatch.context() as patch:
        patch.setattr(Sm100MmaHeuristics, "_select_decode_tile", classmethod(lambda cls, *args: None))
        return _get_config(patch, layer_config, shape_m, **kwargs)


def _assert_legal_on_b200(layer_config, shape_m, config):
    device = DeviceProfile(
        name="B200", sm_version=100, num_sms=148, max_smem_size=Sm100MmaHeuristics.max_smem_size
    )
    problem = TuningProblem(
        layer_config=layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE, device=device
    )
    schedule_fields = (
        "block_shape",
        "warp_shape",
        "use_stream_k",
        "num_stages",
        "num_ctas_per_sm",
        "mma_type",
    )
    candidate = ScheduleCandidate.from_config("decode", {name: config[name] for name in schedule_fields})
    analysis = analyze_candidate(problem, candidate)
    assert analysis.legal, analysis.rejection_reasons


@pytest.mark.parametrize(
    "weights,shape_n,shape_k,shape_m,num_ctas_per_sm,expected_block_k",
    [
        ("int4", 6144, 4096, 1, 1, 256),
        ("int4", 4096, 4096, 8, 1, 256),
        ("int4", 2560, 4096, 1, 1, 256),
        ("int4", 8192, 2048, 32, 1, 256),
        ("int4", 5120, 5120, 32, 1, 256),
        ("int4", 5120, 8192, 16, 1, 256),
        ("int4", 10240, 5120, 16, 2, 256),
        ("int4", 24576, 4096, 8, 3, 256),
        ("int4", 34816, 5120, 1, 4, 256),
        ("fp8", 6144, 4096, 1, 1, 256),
        ("fp8", 5120, 5120, 32, 1, 256),
        ("fp8_block", 4096, 4096, 16, 1, 256),
        ("fp8", 24576, 4096, 8, 3, 128),
        ("fp8_block", 34816, 5120, 1, 4, 128),
        ("nvfp4", 6144, 4096, 1, 1, 256),
        ("nvfp4", 5120, 8192, 16, 1, 256),
        ("nvfp4", 24576, 4096, 16, 3, 256),
        ("nvfp4", 37888, 3584, 1, 4, 256),
    ],
)
def test_sm100_decode_uses_one_wave_data_parallel_tiles(
    monkeypatch, weights, shape_n, shape_k, shape_m, num_ctas_per_sm, expected_block_k
):
    layer_config = _a16_layer_config(shape_n, shape_k, weights)

    config = _get_config(monkeypatch, layer_config, shape_m)

    block_m, block_n, block_k = config["block_shape"]
    num_tiles = shape_n // block_n
    assert config["mma_type"] == "mma" and not config["use_stream_k"]
    assert block_m >= min(shape_m, 8) and block_m <= 32
    assert (block_n, block_k) == (64, expected_block_k)
    assert config["num_ctas_per_sm"] == num_ctas_per_sm
    assert num_tiles <= config["num_sms"] * num_ctas_per_sm <= 148 * num_ctas_per_sm
    _assert_legal_on_b200(layer_config, shape_m, config)
    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        GemmType.DENSE,
        config["num_stages"],
        warp_shape=config["warp_shape"],
    )
    assert smem_size * num_ctas_per_sm <= Sm100MmaHeuristics.max_smem_size


@pytest.mark.parametrize(
    "shape_n,shape_k,shape_m,weights,kwargs",
    [
        (4096, 12288, 1, "int4", {}),  # long K
        (6144, 4096, 48, "int4", {}),  # M above the decode range
        (5120, 8192, 32, "int4", {}),  # K above 5120 at M = 32
        (2048, 6144, 1, "int4", {}),  # too few tiles for this K
        (1280, 8192, 1, "int4", {}),
        (24576, 4096, 32, "int4", {}),  # several CTAs per SM at M = 32
        (51200, 5120, 1, "int4", {}),  # more than four CTAs per SM
        (4096, 4096, 1, "int8", {}),
        (4096, 4096, 1, "int4", {"use_batch_invariant": True}),
        (4096, 4096, 1, "int4", {"use_f16_accum": True}),
        (5120, 8192, 1, "fp8", {}),  # FP8 above K = 5120
        (7168, 8192, 1, "fp8_block", {}),
        (24576, 4096, 16, "fp8", {}),  # FP8 shares SMs between CTAs only up to M = 8
        (37888, 3584, 1, "fp8", {}),  # Stream-K tiles already fill whole waves
        (37888, 3584, 8, "fp8_block", {}),
        (6144, 4096, 1, "fp8_e5m2", {}),  # not measured
        (4096, 12288, 1, "nvfp4", {}),
        (6144, 4096, 48, "nvfp4", {}),
        (6144, 4096, 1, "mxfp4", {}),  # not measured
        (4096, 4096, 1, "fp8", {"use_batch_invariant": True}),
    ],
)
def test_sm100_keeps_heuristic_outside_decode_range(monkeypatch, shape_n, shape_k, shape_m, weights, kwargs):
    layer_config = _a16_layer_config(shape_n, shape_k, weights)

    config = _get_config(monkeypatch, layer_config, shape_m, **kwargs)

    assert config == _get_config_without_decode_tile(monkeypatch, layer_config, shape_m, **kwargs)


@pytest.mark.parametrize("sm_version", [103, 110])
def test_other_sm10x_keep_heuristic(monkeypatch, sm_version):
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: sm_version))
    layer_config = _w4a16_layer_config(6144, 4096, sm_version=sm_version)

    config = _get_config(monkeypatch, layer_config, 1)

    assert config == _get_config_without_decode_tile(monkeypatch, layer_config, 1)


@pytest.mark.parametrize("weights", ["int4", "fp8", "fp8_block", "nvfp4"])
def test_sm100_interval_table_uses_data_parallel_tiles_for_decode(weights):
    layer_config = _a16_layer_config(6144, 4096, weights)

    intervals = Sm100MmaHeuristics.get_configs(layer_config, gemm_type=GemmType.DENSE)

    decode_intervals = [config for _, max_shape_m, config in intervals if max_shape_m <= 32]
    larger_intervals = [config for min_shape_m, _, config in intervals if 32 <= min_shape_m < 256]
    assert decode_intervals and not any(config["use_stream_k"] for config in decode_intervals)
    assert all(config["block_shape"][1:] == (64, 256) for config in decode_intervals)
    assert larger_intervals and all(config["use_stream_k"] for config in larger_intervals)

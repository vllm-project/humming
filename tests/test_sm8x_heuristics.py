import dataclasses

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig
from humming.device import DeviceInfo
from humming.tune.base import DeviceHeuristics
from humming.tune.candidate import DeviceProfile, ScheduleCandidate, TuningProblem, analyze_candidate
from humming.tune.sm8x import Sm80Heuristics, Sm86Heuristics, Sm87Heuristics, Sm89Heuristics
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


@pytest.fixture
def _mock_a100_device(monkeypatch):
    tensorcore_tops = {"float16": 312.0, "bfloat16": 312.0, "int8": 624.0, "int4": 1248.0}
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 108))
    monkeypatch.setattr(DeviceInfo, "sm_version", property(lambda self: 80))
    monkeypatch.setattr(DeviceInfo, "memory_bandwidth_gbps", property(lambda self: 2039.0))
    monkeypatch.setattr(DeviceInfo, "tensorcore_tops", property(lambda self: tensorcore_tops))


def _w4a16_layer_config(shape_n, shape_k, b_dtype=dtypes.uint4, sm_version=80):
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


def _without_evict_hint(config):
    # evict_weights_first is set by Sm80Heuristics.get_config on top of the tile choice.
    return {name: value for name, value in config.items() if name != "evict_weights_first"}


def _assert_legal_on_a100(layer_config, shape_m, config):
    device = DeviceProfile(
        name="A100", sm_version=80, num_sms=108, max_smem_size=Sm80Heuristics.max_smem_size
    )
    problem = TuningProblem(
        layer_config=layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE, device=device
    )
    schedule_fields = ("block_shape", "warp_shape", "use_stream_k", "num_stages", "num_ctas_per_sm")
    candidate = ScheduleCandidate.from_config("decode", {name: config[name] for name in schedule_fields})
    analysis = analyze_candidate(problem, candidate)
    assert analysis.legal, analysis.rejection_reasons


@pytest.mark.usefixtures("_mock_a100_device")
@pytest.mark.parametrize(
    "shape_n,shape_k,shape_m",
    [
        (6144, 4096, 1),
        (4096, 4096, 8),
        (24576, 4096, 16),
        (5120, 5120, 16),
        (6144, 4096, 32),
        (2560, 4096, 4),
    ],
)
def test_sm80_decode_uses_one_wave_data_parallel_tiles(shape_n, shape_k, shape_m):
    layer_config = _w4a16_layer_config(shape_n, shape_k)

    config = Sm80Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE)

    block_m, block_n, _ = config["block_shape"]
    num_tiles = shape_n // block_n
    assert not config["use_stream_k"]
    assert block_m >= min(shape_m, 8) and block_m <= 32
    assert block_n in (64, 128)
    assert num_tiles <= config["num_sms"] * config["num_ctas_per_sm"] <= 108 * config["num_ctas_per_sm"]
    _assert_legal_on_a100(layer_config, shape_m, config)
    smem_size = estimate_smem_size_layer(
        layer_config,
        config["block_shape"],
        GemmType.DENSE,
        config["num_stages"],
        warp_shape=config["warp_shape"],
    )
    assert smem_size * config["num_ctas_per_sm"] <= Sm80Heuristics.max_smem_size


@pytest.mark.usefixtures("_mock_a100_device")
@pytest.mark.parametrize(
    "shape_n,shape_k,shape_m,b_dtype,kwargs",
    [
        (4096, 12288, 1, dtypes.uint4, {}),
        (6144, 4096, 64, dtypes.uint4, {}),
        (5120, 5120, 32, dtypes.uint4, {}),
        (2560, 4096, 32, dtypes.uint4, {}),
        (28672, 4096, 1, dtypes.uint4, {}),
        (4096, 4096, 1, dtypes.int8, {}),
        (4096, 4096, 1, dtypes.uint4, {"use_batch_invariant": True}),
        (4096, 4096, 1, dtypes.uint4, {"use_f16_accum": True}),
    ],
)
def test_sm80_keeps_heuristic_outside_decode_range(shape_n, shape_k, shape_m, b_dtype, kwargs):
    layer_config = _w4a16_layer_config(shape_n, shape_k, b_dtype)

    config = Sm80Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE, **kwargs)
    base_config = DeviceHeuristics.get_config.__func__(
        Sm80Heuristics, layer_config, shape_m, gemm_type=GemmType.DENSE, **kwargs
    )
    assert _without_evict_hint(config) == base_config


@pytest.mark.usefixtures("_mock_a100_device")
def test_sm87_keeps_heuristic():
    layer_config = _w4a16_layer_config(6144, 4096, sm_version=87)

    config = Sm87Heuristics.get_config(layer_config, shape_m=1, gemm_type=GemmType.DENSE)
    base_config = DeviceHeuristics.get_config.__func__(
        Sm87Heuristics, layer_config, 1, gemm_type=GemmType.DENSE
    )
    assert _without_evict_hint(config) == base_config


@pytest.mark.usefixtures("_mock_a100_device")
def test_sm80_interval_table_uses_data_parallel_tiles_for_decode():
    layer_config = _w4a16_layer_config(6144, 4096)

    intervals = Sm80Heuristics.get_configs(layer_config, gemm_type=GemmType.DENSE)

    decode_intervals = [config for _, max_shape_m, config in intervals if max_shape_m <= 32]
    larger_intervals = [config for min_shape_m, _, config in intervals if 32 <= min_shape_m < 4096]
    assert decode_intervals and not any(config["use_stream_k"] for config in decode_intervals)
    assert larger_intervals and all(config["use_stream_k"] for config in larger_intervals)


@pytest.mark.usefixtures("_mock_a100_device")
@pytest.mark.parametrize(
    "num_sms,shape_n,expected_block_n",
    [(108, 6208, 64), (74, 9536, None), (74, 19008, None), (108, 13888, None), (108, 27712, None)],
)
def test_sm80_decode_uses_only_block_n_dividing_n(monkeypatch, num_sms, shape_n, expected_block_n):
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: num_sms))
    layer_config = _w4a16_layer_config(shape_n, 4096)

    config = Sm80Heuristics.get_config(layer_config, shape_m=1, gemm_type=GemmType.DENSE)

    base_config = DeviceHeuristics.get_config.__func__(
        Sm80Heuristics, layer_config, 1, gemm_type=GemmType.DENSE
    )
    if expected_block_n is None:
        assert _without_evict_hint(config) == base_config
        return
    assert not config["use_stream_k"] and config["block_shape"][1] == expected_block_n
    assert config["num_sms"] * config["num_ctas_per_sm"] >= shape_n // expected_block_n


@pytest.mark.usefixtures("_mock_a100_device")
@pytest.mark.parametrize("heuristics_cls,sm_version", [(Sm80Heuristics, 80), (Sm87Heuristics, 87)])
@pytest.mark.parametrize(
    "shape_n,shape_k,shape_m,should_evict",
    [
        (6144, 4096, 1, True),  # SM80 decode tile
        (4096, 12288, 1, True),  # long K keeps the default heuristic
        (6144, 4096, 4096, False),
    ],
)
def test_sm8x_dense_evicts_weights_when_one_m_block(
    heuristics_cls, sm_version, shape_n, shape_k, shape_m, should_evict
):
    layer_config = _w4a16_layer_config(shape_n, shape_k, sm_version=sm_version)

    config = heuristics_cls.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE)

    assert config["evict_weights_first"] == should_evict
    assert (shape_m <= config["block_shape"][0]) == should_evict


def _base_sm80_config(layer_config, shape_m):
    return DeviceHeuristics.get_config.__func__(
        Sm80Heuristics, layer_config, shape_m, gemm_type=GemmType.DENSE
    )


@pytest.mark.usefixtures("_mock_a100_device")
def test_sm80_stream_k_longest_iters_matches_scheduler():
    # 34816 x 5120, block (8, 256, 64), 216 CTAs: 136 tiles x 80 K blocks over 216 CTAs is 51 iterations,
    # rounded up to lcm(2, 5) = 10 with 5 stages and to lcm(2, 4) = 4 with 4.
    layer_config = _w4a16_layer_config(34816, 5120, b_dtype=dtypes.uint3)
    config = _base_sm80_config(layer_config, 1)
    assert config["block_shape"] == (8, 256, 64) and config["num_sms"] * config["num_ctas_per_sm"] == 216
    assert Sm80Heuristics._stream_k_longest_iters(layer_config, config, 1, 5) == (60, False)
    assert Sm80Heuristics._stream_k_longest_iters(layer_config, config, 1, 4) == (52, False)


@pytest.mark.usefixtures("_mock_a100_device")
@pytest.mark.parametrize(
    "shape_n,shape_k,shape_m",
    [
        (34816, 5120, 1),  # 60 -> 52 iterations
        (34816, 5120, 16),
        (28672, 4096, 8),
        (17408, 5120, 1),
        (24576, 4096, 1),  # 4 stages makes every slice one whole tile
    ],
)
def test_sm80_int3_stream_k_drops_unbalancing_stage(shape_n, shape_k, shape_m):
    layer_config = _w4a16_layer_config(shape_n, shape_k, b_dtype=dtypes.uint3)

    config = Sm80Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE)
    base_config = _base_sm80_config(layer_config, shape_m)

    assert base_config["use_stream_k"] and base_config["num_stages"] == 5
    assert _without_evict_hint(config) == dict(base_config, num_stages=4)


@pytest.mark.usefixtures("_mock_a100_device")
@pytest.mark.parametrize(
    "shape_n,shape_k,shape_m,b_dtype",
    [
        (20480, 5120, 1, dtypes.uint3),  # 4 stages would round 30 iterations up to 32
        (40960, 5120, 1, dtypes.uint3),  # same iterations either way
        (34816, 5120, 1, dtypes.uint4),  # INT4 is bandwidth-bound there; rebalancing barely helps
        (34816, 5120, 1, dtypes.uint8),
    ],
)
def test_sm80_stream_k_keeps_stages_otherwise(shape_n, shape_k, shape_m, b_dtype):
    layer_config = _w4a16_layer_config(shape_n, shape_k, b_dtype=b_dtype)

    config = Sm80Heuristics.get_config(layer_config, shape_m=shape_m, gemm_type=GemmType.DENSE)

    assert _without_evict_hint(config) == _base_sm80_config(layer_config, shape_m)


@pytest.mark.usefixtures("_mock_a100_device")
def test_sm80_stream_k_empty_tail_keeps_stages(monkeypatch):
    # 140 tiles on 70 CTAs: two data-parallel tiles each and no Stream-K tail, whatever the stage count.
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 70))
    layer_config = _w4a16_layer_config(35840, 2048, b_dtype=dtypes.uint3)
    config = _base_sm80_config(layer_config, 32)
    assert config["block_shape"] == (32, 256, 64) and config["num_ctas_per_sm"] == 1
    assert config["use_stream_k"] and config["num_stages"] == 5
    for num_stages in (4, 5):
        assert Sm80Heuristics._stream_k_longest_iters(layer_config, config, 32, num_stages) == (64, True)
    assert not Sm80Heuristics._should_drop_stream_k_stage(layer_config, config, 32, GemmType.DENSE)

import math

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, TuningConfig
from humming.config.mma import get_register_budget_error
from humming.device import DeviceInfo
from humming.tune import _get_heuristics_config
from humming.tune.sm90 import Sm90Heuristics
from humming.tune.sm90_h20 import Sm90H20Heuristics
from humming.tune.sm90_policies import get_small_m_packed_fp4_config


@pytest.fixture(autouse=True)
def _mock_h200_sm_count(monkeypatch):
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: 132))


def _layer(
    shape_n: int,
    shape_k: int,
    *,
    num_experts: int = 12,
    a_dtype=dtypes.float8e4m3,
    as_dtype=dtypes.float32,
    bs_dtype=dtypes.float8e8m0,
    input_scale_group_size: int = 128,
    weight_scale_group_size: int = 32,
    use_packed_k_layout: bool | None = None,
) -> LayerConfig:
    return LayerConfig(
        sm_version=90,
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=a_dtype,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        as_dtype=as_dtype,
        bs_dtype=bs_dtype,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=weight_scale_group_size,
        use_packed_k_layout=use_packed_k_layout,
    )


@pytest.mark.parametrize(
    ("layer", "gemm_type"),
    [
        (
            _layer(
                2880,
                2880,
                input_scale_group_size=32,
                as_dtype=dtypes.float32,
            ),
            GemmType.DENSE,
        ),
        (
            _layer(
                5760,
                2880,
                num_experts=32,
                a_dtype=dtypes.bfloat16,
                as_dtype=None,
                input_scale_group_size=0,
            ),
            GemmType.INDEXED,
        ),
    ],
)
def test_migrated_policies_own_proposal_generation(
    monkeypatch,
    layer,
    gemm_type,
):
    def fail_legacy_helper(cls, *args, **kwargs):
        raise AssertionError("migrated policy called a legacy proposal helper")

    monkeypatch.setattr(
        Sm90Heuristics,
        "get_config1",
        classmethod(fail_legacy_helper),
    )
    monkeypatch.setattr(
        Sm90Heuristics,
        "calc_num_block_list",
        classmethod(fail_legacy_helper),
    )

    decision = Sm90Heuristics.get_tuning_decision(
        layer,
        shape_m=32,
        gemm_type=gemm_type,
    )

    assert decision.selected_analysis.legal


def test_grouped_and_legacy_selection_work_without_a_live_device(monkeypatch):
    def fail_device_query(self):
        raise AssertionError("unexpected live device query")

    monkeypatch.setattr(DeviceInfo, "sm_count", property(fail_device_query))

    grouped = Sm90Heuristics.get_config(
        _layer(6144, 3584, num_experts=0, use_packed_k_layout=False),
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )
    legacy_a16 = Sm90Heuristics.get_config(
        _layer(
            5760,
            2880,
            num_experts=0,
            a_dtype=dtypes.bfloat16,
            as_dtype=None,
            input_scale_group_size=0,
        ),
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )

    assert grouped["block_shape"] == (32, 128, 256)
    assert legacy_a16["block_shape"][1:] == (128, 64)


@pytest.mark.parametrize(
    ("input_scale_group_size", "as_dtype"),
    [(128, dtypes.float32), (32, dtypes.float32)],
)
def test_grouped_fp8_moe_avoids_512_thread_tiles(
    input_scale_group_size,
    as_dtype,
):
    config = Sm90Heuristics.get_config(
        _layer(
            6144,
            3584,
            input_scale_group_size=input_scale_group_size,
            as_dtype=as_dtype,
        ),
        shape_m=64,
        gemm_type=GemmType.INDEXED,
    )

    assert config["warp_shape"][1] == 32
    block_volume = math.prod(config["block_shape"])
    warp_volume = math.prod(config["warp_shape"])
    num_threads = block_volume // warp_volume * 32
    assert num_threads <= 256


def test_grouped_fp8_uses_legal_n_tile_for_non_128_multiple():
    decision = Sm90Heuristics.get_tuning_decision(
        _layer(
            2880,
            2880,
            input_scale_group_size=32,
            as_dtype=dtypes.float32,
        ),
        shape_m=32,
        gemm_type=GemmType.DENSE,
    )
    config = decision.to_config()

    assert decision.selected_analysis.legal
    assert "multi_cast_size_a" not in config
    assert config["block_shape"][1] == 64
    assert config["warp_shape"][1] == 16
    assert config["block_shape"][2] == 64
    assert config["warp_shape"][2] == 64


def test_grouped_fp8_uses_legal_schedules_across_the_dense_threshold():
    layer = _layer(6144, 3584, num_experts=0, use_packed_k_layout=False)

    below_threshold = Sm90Heuristics.get_config(
        layer,
        shape_m=504,
        gemm_type=GemmType.DENSE,
    )
    at_threshold = Sm90Heuristics.get_config(
        layer,
        shape_m=512,
        gemm_type=GemmType.DENSE,
    )

    for config in (below_threshold, at_threshold):
        tuning = TuningConfig(**config)
        assert get_register_budget_error(layer, tuning, registers_per_sm=65536) is None


def test_dense_a16_uses_legal_n_tile_for_non_256_multiple():
    config = Sm90Heuristics.get_config(
        _layer(
            5760,
            2880,
            a_dtype=dtypes.bfloat16,
            as_dtype=None,
            input_scale_group_size=0,
        ),
        shape_m=16,
        gemm_type=GemmType.DENSE,
    )

    assert config["block_shape"][1] == 128
    assert config["warp_shape"][1] == 32


def test_dense_a16_requires_padding_when_n_has_no_wgmma_tile():
    unpadded = _layer(
        2880,
        2880,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        input_scale_group_size=0,
    )
    padded = _layer(
        3072,
        3072,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        input_scale_group_size=0,
    )

    with pytest.raises(AssertionError):
        Sm90Heuristics.get_config(
            unpadded,
            shape_m=16,
            gemm_type=GemmType.DENSE,
        )

    config = Sm90Heuristics.get_config(
        padded,
        shape_m=16,
        gemm_type=GemmType.DENSE,
    )
    assert config["block_shape"][1:] == (128, 64)
    assert config["warp_shape"][1:] == (32, 64)


def test_short_k_does_not_leave_warp_k_larger_than_block_k():
    config = Sm90Heuristics.get_config(
        LayerConfig(
            sm_version=90,
            shape_n=128,
            shape_k=64,
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint7,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
        ),
        shape_m=64,
        gemm_type=GemmType.DENSE,
    )

    assert config["block_shape"][2] == 64
    assert config["warp_shape"][2] == 64


@pytest.mark.parametrize(
    ("routed_m", "expected_m", "expected_ctas", "expected_stream_k"),
    [(2, 8, 1, True), (4, 8, 2, False), (8, 8, 3, False), (256, 16, 2, False)],
)
def test_mxfp4_a16_indexed_preserves_warp_k_and_fits_grid(
    routed_m,
    expected_m,
    expected_ctas,
    expected_stream_k,
):
    config = Sm90Heuristics.get_config(
        _layer(
            5760,
            2880,
            num_experts=32,
            a_dtype=dtypes.bfloat16,
            as_dtype=None,
            input_scale_group_size=0,
        ),
        shape_m=routed_m,
        gemm_type=GemmType.INDEXED,
    )

    assert config["block_shape"] == (expected_m, 128, 64)
    assert config["warp_shape"] == (expected_m, 32, 64)
    assert config["num_ctas_per_sm"] == expected_ctas
    assert config["use_stream_k"] is expected_stream_k


@pytest.mark.parametrize("shape_k", [512, 256])
@pytest.mark.parametrize(
    ("shape_m", "block_m", "block_n", "num_ctas"),
    [(6144, 32, 512, 2), (8192, 40, 512, 1), (10240, 48, 256, 2), (12288, 64, 256, 2)],
)
def test_mxfp4_a16_short_k_limits_wide_n_tile_before_block_m48(shape_k, shape_m, block_m, block_n, num_ctas):
    layer = _layer(
        6144,
        shape_k,
        num_experts=256,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        input_scale_group_size=0,
    )

    config = Sm90Heuristics.get_config(
        layer,
        shape_m=shape_m,
        gemm_type=GemmType.INDEXED,
    )

    assert config["block_shape"] == (block_m, block_n, 64)
    assert config["num_ctas_per_sm"] == num_ctas
    tuning_fields = TuningConfig.get_field_names()
    tuning = TuningConfig(**{key: value for key, value in config.items() if key in tuning_fields})
    assert get_register_budget_error(layer, tuning, registers_per_sm=65536) is None


@pytest.mark.parametrize(
    ("shape_n", "shape_k", "num_experts"),
    [
        (4608, 5120, 384),  # DeepSeek-V4.1-Flash W13
        (5120, 2304, 384),  # DeepSeek-V4.1-Flash W2
        (6144, 3584, 896),  # Kimi-K3 W13
        (3584, 3072, 896),  # Kimi-K3 W2
    ],
)
def test_small_m_grouped_fp4_a8_uses_wide_non_stream_k_tile(
    shape_n,
    shape_k,
    num_experts,
):
    layer = _layer(
        shape_n,
        shape_k,
        num_experts=num_experts,
        input_scale_group_size=32,
    )
    config = get_small_m_packed_fp4_config(
        layer,
        shape_m=2 * num_experts,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
        use_f16_accum=False,
        use_batch_invariant=False,
    )

    assert config == {
        "mma_type": "wgmma",
        "block_shape": (8, 256, 128),
        "warp_shape": (8, 32, 128),
        "num_stages": 4,
        "num_ctas_per_sm": 2,
        "use_stream_k": False,
        "use_tma": True,
        "use_warp_spec": True,
    }


@pytest.mark.parametrize(
    ("shape_n", "shape_k", "num_experts", "block_n", "num_ctas"),
    [
        (4608, 5120, 384, 128, 4),
        (5120, 2304, 384, 256, 2),
        (6144, 3584, 896, 128, 4),
        (3584, 3072, 896, 128, 4),
    ],
)
def test_small_m_grouped_fp4_a16_uses_mma_without_stream_k(
    shape_n,
    shape_k,
    num_experts,
    block_n,
    num_ctas,
):
    layer = _layer(
        shape_n,
        shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        input_scale_group_size=0,
    )
    config = get_small_m_packed_fp4_config(
        layer,
        shape_m=2 * num_experts,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
        use_f16_accum=False,
        use_batch_invariant=False,
    )

    assert config == {
        "mma_type": "mma",
        "block_shape": (8, block_n, 128),
        "warp_shape": (8, 64, 64),
        "num_stages": 4,
        "num_ctas_per_sm": num_ctas,
        "use_stream_k": False,
        "use_tma": False,
        "use_warp_spec": False,
    }


def test_small_m_grouped_fp4_policy_is_narrowly_scoped():
    a8_layer = _layer(4608, 5120, num_experts=384, input_scale_group_size=32)
    kwargs = {
        "gemm_type": GemmType.GROUPED_CONTIGUOUS,
        "use_f16_accum": False,
        "use_batch_invariant": False,
    }
    indexed_kwargs = kwargs | {"gemm_type": GemmType.INDEXED}
    f16_accum_kwargs = kwargs | {"use_f16_accum": True}

    assert (
        get_small_m_packed_fp4_config(
            a8_layer,
            shape_m=512,
            **kwargs,
        )
        is None
    )
    assert (
        get_small_m_packed_fp4_config(
            a8_layer,
            shape_m=8 * a8_layer.num_experts + 8,
            **kwargs,
        )
        is None
    )
    assert (
        get_small_m_packed_fp4_config(
            a8_layer,
            shape_m=768,
            **indexed_kwargs,
        )
        is None
    )
    assert (
        get_small_m_packed_fp4_config(
            a8_layer,
            shape_m=768,
            **f16_accum_kwargs,
        )
        is None
    )


def test_nvfp4_a16_uses_narrow_first_wave_only():
    layer = _layer(
        5376,
        2688,
        num_experts=128,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        bs_dtype=dtypes.float8e4m3,
        input_scale_group_size=0,
        weight_scale_group_size=16,
    )
    below_first_wave = Sm90Heuristics.get_config(
        layer,
        shape_m=4,
        gemm_type=GemmType.INDEXED,
    )
    first_wave = Sm90Heuristics.get_config(
        layer,
        shape_m=8,
        gemm_type=GemmType.INDEXED,
    )
    next_wave = Sm90Heuristics.get_config(
        layer,
        shape_m=16,
        gemm_type=GemmType.INDEXED,
    )

    assert below_first_wave["block_shape"] == (8, 256, 128)
    assert below_first_wave["warp_shape"] == (8, 32, 64)
    assert below_first_wave["num_ctas_per_sm"] == 1
    assert below_first_wave["use_stream_k"] is True
    assert first_wave["block_shape"] == (8, 128, 128)
    assert first_wave["warp_shape"] == (8, 32, 64)
    assert first_wave["num_ctas_per_sm"] == 3
    assert next_wave["block_shape"] == (8, 256, 128)
    assert next_wave["warp_shape"] == (8, 64, 64)


def test_fp4_a16_dense_uses_small_tile_only_through_m128():
    layer = _layer(
        5376,
        2688,
        num_experts=0,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        bs_dtype=dtypes.float8e4m3,
        input_scale_group_size=0,
        weight_scale_group_size=16,
    )
    small = Sm90Heuristics.get_config(layer, shape_m=128, gemm_type=GemmType.DENSE)
    large = Sm90Heuristics.get_config(layer, shape_m=256, gemm_type=GemmType.DENSE)

    assert small["block_shape"][1:] == (128, 64)
    assert small["warp_shape"][1:] == (32, 64)
    assert small["num_ctas_per_sm"] == 2
    assert large["block_shape"][1:] != (128, 64)


@pytest.mark.parametrize("device_name", ["H200", "L20X"])
@pytest.mark.parametrize("num_experts", [8, 33])
@pytest.mark.parametrize("shape_n,shape_k", [(4096, 6144), (6144, 2048), (4096, 3072), (4096, 4096)])
def test_grouped_w4a8_ranges_match_direct_selection(monkeypatch, device_name, num_experts, shape_n, shape_k):
    monkeypatch.setattr(DeviceInfo, "name", property(lambda self: device_name))
    monkeypatch.setattr("humming.tune.get_heuristics_class", lambda **kwargs: Sm90Heuristics)
    layer = _layer(shape_n, shape_k, num_experts=num_experts)
    assert layer.use_packed_k_layout
    select = _get_heuristics_config.__wrapped__
    kwargs = dict(use_m_major_input_scale=True, gemm_type=GemmType.GROUPED_CONTIGUOUS)
    ranges = select(layer, **kwargs)
    assert ranges[0][0] == 0 and ranges[-1][1] == 1 << 30
    previous_upper = 0
    for lower, upper, config in ranges:
        assert lower == previous_upper and lower < upper
        assert config["warp_shape"][2] == 128
        assert config["wgmma_split_issue_wait"]
        tuning_keys = (
            "mma_type",
            "block_shape",
            "warp_shape",
            "num_stages",
            "use_warp_spec",
            "num_ctas_per_sm",
        )
        tuning = TuningConfig(
            **{key: config[key] for key in tuning_keys},
            wgmma_split_issue_wait=config["wgmma_split_issue_wait"],
        )
        assert get_register_budget_error(layer, tuning, registers_per_sm=65536) is None
        for shape_m in (lower + 1, min(upper, lower + num_experts * 2048)):
            assert select(layer, shape_m=shape_m, **kwargs) == config
        previous_upper = upper


@pytest.mark.parametrize("gemm_type", [GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS])
@pytest.mark.parametrize("routed_m", [8, 256, 1024])
def test_small_a16_expert_tiles_use_mma_without_padding(gemm_type, routed_m):
    layer = _layer(
        1024,
        7168,
        num_experts=256,
        a_dtype=dtypes.bfloat16,
        as_dtype=None,
        bs_dtype=dtypes.float8e4m3,
        input_scale_group_size=0,
        weight_scale_group_size=16,
    )
    config = Sm90Heuristics.get_config(layer, routed_m, gemm_type=gemm_type)

    assert config["mma_type"] == "mma"
    assert config["block_shape"][0] == config["warp_shape"][0] == 8
    assert config["use_stream_k"] is True
    assert not config.get("use_tma", False)
    assert not config.get("use_warp_spec", False)


@pytest.mark.parametrize("num_experts", [64, 256])
def test_token_scaled_grouped_selects_residency_from_expert_tiles(num_experts):
    layer = _layer(1536, 3072, num_experts=num_experts, input_scale_group_size=0)
    config = Sm90Heuristics.get_config(
        layer, shape_m=256, gemm_type=GemmType.GROUPED_CONTIGUOUS, use_m_major_input_scale=True
    )
    assert not layer.use_packed_k_layout
    assert not config["use_warp_spec"]
    assert not config["use_tma"]
    assert config["num_ctas_per_sm"] == (3 if num_experts == 64 else 2)
    tuning_values = {key: value for key, value in config.items() if key in TuningConfig.get_field_names()}
    tuning = TuningConfig(**tuning_values)
    assert get_register_budget_error(layer, tuning, registers_per_sm=65536) is None


def test_token_scaled_grouped_keeps_overlap_for_short_grid():
    layer = _layer(4096, 7168, num_experts=256, input_scale_group_size=0)
    config = Sm90Heuristics.get_config(layer, shape_m=8, gemm_type=GemmType.GROUPED_CONTIGUOUS)
    assert config["num_ctas_per_sm"] == 1
    assert config["use_warp_spec"]


def test_token_scaled_grouped_can_reuse_weights_across_large_m_tile():
    layer = _layer(4096, 7168, num_experts=256, input_scale_group_size=0)
    config = Sm90Heuristics.get_config(
        layer, shape_m=32768, gemm_type=GemmType.GROUPED_CONTIGUOUS, use_m_major_input_scale=True
    )
    assert config["block_shape"][0] > 128
    assert config["use_warp_spec"]
    assert not config["wgmma_use_late_as"]
    tuning_values = {key: value for key, value in config.items() if key in TuningConfig.get_field_names()}
    tuning = TuningConfig(**tuning_values)
    assert get_register_budget_error(layer, tuning, registers_per_sm=65536) is None


@pytest.mark.parametrize("input_scale_group_size", [0, 128])
@pytest.mark.parametrize("shape_n,shape_k", [(1536, 3072), (2816, 1792), (3072, 6144)])
@pytest.mark.parametrize("routed_m", [192, 2560, 32768])
def test_indexed_a8_residency_fits_registers(input_scale_group_size, shape_n, shape_k, routed_m):
    layer = _layer(shape_n, shape_k, num_experts=64, input_scale_group_size=input_scale_group_size)
    config = Sm90Heuristics.get_config(layer, routed_m, gemm_type=GemmType.INDEXED)
    tuning_values = {key: value for key, value in config.items() if key in TuningConfig.get_field_names()}
    tuning = TuningConfig(**tuning_values)
    assert get_register_budget_error(layer, tuning, registers_per_sm=65536) is None


def test_indexed_token_scale_keeps_residency_when_an_extra_wave_dominates():
    layer = _layer(1536, 3072, num_experts=64, input_scale_group_size=0)
    config = Sm90Heuristics.get_config(layer, shape_m=256, gemm_type=GemmType.INDEXED)
    assert config["num_ctas_per_sm"] == 1
    assert config["use_warp_spec"]


@pytest.mark.parametrize("heuristics,num_sms", [(Sm90Heuristics, 132), (Sm90H20Heuristics, 78)])
@pytest.mark.parametrize(
    "gemm_type", [GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED]
)
@pytest.mark.parametrize("input_scale_group_size", [0, 128])
@pytest.mark.parametrize("shape_n", [4096, 4224])
@pytest.mark.parametrize("routed_m,use_stream_k", [(8, True), (65536, False)])
def test_long_k_moe_stream_k_depends_on_grid_size(
    monkeypatch, heuristics, num_sms, gemm_type, input_scale_group_size, shape_n, routed_m, use_stream_k
):
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: num_sms))
    layer = _layer(shape_n, 7168, num_experts=256, input_scale_group_size=input_scale_group_size)
    config = heuristics.get_config(layer, routed_m, gemm_type=gemm_type, use_m_major_input_scale=True)
    assert config["use_stream_k"] is use_stream_k


@pytest.mark.parametrize("heuristics,num_sms", [(Sm90Heuristics, 132), (Sm90H20Heuristics, 78)])
@pytest.mark.parametrize("gemm_type", [GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS])
@pytest.mark.parametrize("shape_n", [4096, 4224])
@pytest.mark.parametrize("routed_m", [16384, 65536])
def test_large_token_scaled_moe_considers_wider_m_tiles(
    monkeypatch, heuristics, num_sms, gemm_type, shape_n, routed_m
):
    monkeypatch.setattr(DeviceInfo, "sm_count", property(lambda self: num_sms))
    layer = _layer(shape_n, 2048, num_experts=256, input_scale_group_size=0)
    config = heuristics.get_config(layer, routed_m, gemm_type=gemm_type, use_m_major_input_scale=True)
    assert config["block_shape"][0] > 64
    assert not config["use_stream_k"]
    tuning_values = {key: value for key, value in config.items() if key in TuningConfig.get_field_names()}
    assert get_register_budget_error(layer, TuningConfig(**tuning_values), registers_per_sm=65536) is None

from types import SimpleNamespace

import pytest

import humming.tune.process_input as tune
from humming import dtypes
from humming.config import ActivationType, ProcessInputLayoutType, ProcessInputProblemConfig

H200 = SimpleNamespace(
    sm_version=90,
    sm_major=9,
    sm_count=132,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    default_smem_size=49152,
    max_smem_size=232448,
    l2_cache_size=62914560,
)
B300 = SimpleNamespace(
    sm_version=103,
    sm_major=10,
    sm_count=148,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    default_smem_size=49152,
    max_smem_size=232448,
    l2_cache_size=132644864,
)
GB10 = SimpleNamespace(
    sm_version=121,
    sm_major=12,
    sm_count=48,
    max_threads_per_sm=1536,
    max_threads_per_block=1024,
    default_smem_size=49152,
    max_smem_size=101376,
    l2_cache_size=25165824,
)

SILU_AND_MUL = "(a / (1.f + expf(-a))) * b"
QUANT = {
    "fp8_token": dict(quant_mode="dynamic_token", quant_dtype=dtypes.float8e4m3),
    "fp8_group128": dict(
        quant_mode="dynamic_group",
        quant_dtype=dtypes.float8e4m3,
        quant_group_size=128,
        group_scale_dtype=dtypes.float32,
    ),
    "mxfp8": dict(
        quant_mode="dynamic_group",
        quant_dtype=dtypes.float8e4m3,
        quant_group_size=32,
        group_scale_dtype=dtypes.float8e8m0,
    ),
}


def _select(monkeypatch, device, quant, hidden_size, num_rows, activation):
    monkeypatch.setattr(tune, "DeviceInfo", lambda index: device)
    tune._tuning_config_for_device.cache_clear()
    config = ProcessInputProblemConfig(
        input_dtype=dtypes.bfloat16,
        hidden_size=hidden_size,
        activation_type=ActivationType.BinarySplit if activation else ActivationType.None_,
        activation_impl=SILU_AND_MUL if activation else None,
        layout=ProcessInputLayoutType.Normal,
        **QUANT[quant],
    )
    tuning_config = tune._tuning_config_for_device(config, num_rows, 0)
    tune._tuning_config_for_device.cache_clear()
    return (
        tuning_config.values_per_thread,
        tuning_config.threads_per_task,
        tuning_config.tokens_per_block,
    )


# (values_per_thread, threads_per_task, tokens_per_block)
@pytest.mark.parametrize(
    ("device", "quant", "hidden_size", "num_rows", "activation", "expected"),
    [
        # Activation + group quant on large grids: 16 values per thread, one block per row.
        (H200, "fp8_group128", 1152, 98304, True, (16, 96, 1)),
        (B300, "mxfp8", 2304, 32768, True, (16, 160, 1)),
        # Per-token quant on large grids: one token per block.
        (H200, "fp8_token", 640, 49152, True, (16, 64, 1)),
        (B300, "fp8_token", 2880, 32768, True, (16, 192, 1)),
        # Per-token quant without an activation: 32 values per thread.
        (H200, "fp8_token", 7168, 8192, False, (32, 224, 1)),
        (B300, "fp8_token", 2880, 8192, False, (32, 96, 1)),
        # Group quant without an activation: vector width before idle lanes.
        (B300, "mxfp8", 2880, 8192, False, (32, 96, 1)),
    ],
)
def test_large_grids_use_wide_vectors(
    monkeypatch, device, quant, hidden_size, num_rows, activation, expected
):
    assert _select(monkeypatch, device, quant, hidden_size, num_rows, activation) == expected


@pytest.mark.parametrize(
    ("device", "quant", "hidden_size", "num_rows", "activation", "expected"),
    [
        (H200, "fp8_group128", 1152, 768, True, (8, 128, 1)),
        (B300, "mxfp8", 2048, 512, True, (4, 128, 1)),
        (B300, "fp8_token", 2048, 512, True, (16, 128, 4)),
    ],
)
def test_small_grids_keep_existing_selection(
    monkeypatch, device, quant, hidden_size, num_rows, activation, expected
):
    assert _select(monkeypatch, device, quant, hidden_size, num_rows, activation) == expected


@pytest.mark.parametrize(
    ("device", "quant", "hidden_size", "num_rows", "activation", "expected"),
    [
        # 32 values per thread is 1.4-2x slower on GB10 for activation-free quant.
        (GB10, "fp8_token", 2880, 8192, False, (4, 736, 1)),
        (GB10, "mxfp8", 2880, 8192, False, (4, 736, 1)),
        (GB10, "fp8_token", 2880, 4096, True, (16, 192, 4)),
        (GB10, "mxfp8", 2304, 16384, True, (8, 128, 1)),
    ],
)
def test_sm12x_keeps_narrow_vectors_on_large_grids(
    monkeypatch, device, quant, hidden_size, num_rows, activation, expected
):
    assert _select(monkeypatch, device, quant, hidden_size, num_rows, activation) == expected

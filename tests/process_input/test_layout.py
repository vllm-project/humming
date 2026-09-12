import pytest
import torch

from humming.config import ActivationType, InputQuantizationMode
from humming.ops.input import process_input
from humming.testing.process_input import (
    ACTIVATION_TYPE_IMPL_TEST_MAP,
    assert_process_input_close,
    process_input_ref,
    skip_if_process_input_unsupported,
)


@pytest.mark.parametrize(
    "quant_mode,group_scale_dtype,use_m_major_input_scale",
    [
        ("none", "float32", False),
        ("static_tensor", "float32", False),
        ("dynamic_token", "float32", False),
        ("dynamic_group", "float32", False),
        ("dynamic_group", "float32", True),
        ("dynamic_group", "float8e8m0", True),
        ("static_tensor_dynamic_group", "float8e4m3", True),
        ("dynamic_group_token", "float8e4m3", False),
        ("dynamic_group_token", "float8e4m3", True),
    ],
)
def test_scatter_layout(quant_mode, group_scale_dtype, use_m_major_input_scale):
    quant_mode = InputQuantizationMode(quant_mode)
    quant_dtype = "int8" if quant_mode.should_quantize else None
    skip_if_process_input_unsupported(quant_dtype, group_scale_dtype)
    torch.manual_seed(0)

    inputs = torch.randn(4, 768, device="cuda")
    scatter_idx = torch.tensor([[5, -1, 1], [3, 0, -1], [4, -1, 2], [-1, -1, -1]], device="cuda")
    valid_rows = scatter_idx[scatter_idx >= 0]
    static_tensor_scale = None

    if quant_mode.has_tensor_scale:
        static_tensor_scale = torch.tensor([0.5], device="cuda")

    options = dict(
        quant_mode=quant_mode.value,
        quant_dtype=quant_dtype,
        quant_group_size=128,
        group_scale_dtype=group_scale_dtype,
        layout="scatter",
        scatter_idx=scatter_idx,
        use_m_major_input_scale=use_m_major_input_scale,
    )

    expected = process_input_ref(inputs, static_tensor_scale=static_tensor_scale, **options)
    actual = process_input(inputs, token_scales=static_tensor_scale, **options)

    assert_process_input_close(
        actual,
        expected,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=128,
        use_m_major_input_scale=use_m_major_input_scale,
        valid_rows=valid_rows,
    )


@pytest.mark.parametrize("zero_invalid", [False, True])
@pytest.mark.parametrize("quant_mode", ["none", "dynamic_group", "dynamic_token", "dynamic_group_token"])
@pytest.mark.parametrize("activation_type", ["none", "unary", "binary_split", "binary_interleaved"])
def test_grouped_mask_layout(quant_mode, zero_invalid, activation_type):
    quant_mode = InputQuantizationMode(quant_mode)
    quant_dtype = "int8" if quant_mode.should_quantize else None
    group_scale_dtype = "float8e4m3" if quant_mode == InputQuantizationMode.DynamicGroupToken else "float32"
    skip_if_process_input_unsupported(quant_dtype, group_scale_dtype)
    torch.manual_seed(0)

    inputs = torch.randn(14, 256, device="cuda")
    expert_layout = torch.tensor([3, 1], device="cuda", dtype=torch.int64)
    rows_per_expert = inputs.size(0) // expert_layout.numel()
    local_rows = torch.arange(rows_per_expert, device="cuda")
    valid_rows = (local_rows[None, :] < expert_layout[:, None]).flatten()

    activation_impl = ACTIVATION_TYPE_IMPL_TEST_MAP[ActivationType(activation_type)]["impl"]
    options = dict(
        quant_mode=quant_mode.value,
        quant_dtype=quant_dtype,
        quant_group_size=128,
        group_scale_dtype=group_scale_dtype,
        activation_type=activation_type,
        layout="grouped_mask",
        expert_layout=expert_layout,
        zero_invalid=zero_invalid,
    )

    expected = process_input_ref(inputs, **options)
    actual = process_input(inputs, activation_impl=activation_impl, **options)

    assert_process_input_close(
        actual,
        expected,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=128,
        valid_rows=valid_rows,
        zero_invalid=zero_invalid,
    )

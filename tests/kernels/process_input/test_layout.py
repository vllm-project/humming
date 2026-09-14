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
@pytest.mark.parametrize("zero_invalid", [False, True])
def test_scatter_layout(quant_mode, group_scale_dtype, use_m_major_input_scale, zero_invalid):
    quant_mode = InputQuantizationMode(quant_mode)
    quant_dtype = "int8" if quant_mode.should_quantize else None
    skip_if_process_input_unsupported(quant_dtype, group_scale_dtype)
    torch.manual_seed(0)

    inputs = torch.randn(4, 768, device="cuda")
    index_dtype = torch.int32
    if zero_invalid:
        index_dtype = torch.int64
    scatter_idx = torch.tensor(
        [[5, -1, 1], [3, 0, 12], [4, 20, 2], [-1, -1, -1]],
        device="cuda",
        dtype=index_dtype,
    )
    num_valid_tokens = torch.tensor([4], device="cuda", dtype=index_dtype)
    in_range = (scatter_idx >= 0) & (scatter_idx < scatter_idx.numel())
    valid_rows = scatter_idx[in_range].long()
    if not zero_invalid:
        valid_rows = valid_rows[valid_rows < num_valid_tokens]

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
        num_valid_tokens=num_valid_tokens,
        zero_invalid=zero_invalid,
        use_m_major_input_scale=use_m_major_input_scale,
    )

    expected = process_input_ref(inputs, static_tensor_scale=static_tensor_scale, **options)
    actual = process_input(inputs, token_scales=static_tensor_scale, **options)

    assert_process_input_close(
        actual,
        expected,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        group_scale_dtype=group_scale_dtype,
        quant_group_size=128,
        use_m_major_input_scale=use_m_major_input_scale,
        valid_rows=valid_rows,
    )

    if zero_invalid:
        zero_rows = scatter_idx[in_range & (scatter_idx >= num_valid_tokens)].long()
        output_bytes = actual[0].view(torch.uint8)
        assert torch.count_nonzero(output_bytes[zero_rows]) == 0


@pytest.mark.parametrize("zero_invalid", [False, True])
@pytest.mark.parametrize("quant_mode", ["none", "dynamic_group", "dynamic_token", "dynamic_group_token"])
@pytest.mark.parametrize("activation_type", ["none", "unary", "binary_split", "binary_interleaved"])
@pytest.mark.parametrize("layout", ["normal", "grouped_mask"])
def test_masked_layout(quant_mode, zero_invalid, activation_type, layout):
    quant_mode = InputQuantizationMode(quant_mode)
    quant_dtype = "int8" if quant_mode.should_quantize else None
    group_scale_dtype = "float8e4m3" if quant_mode == InputQuantizationMode.DynamicGroupToken else "float32"
    skip_if_process_input_unsupported(quant_dtype, group_scale_dtype)
    torch.manual_seed(0)

    inputs = torch.randn(14, 256, device="cuda")
    expert_tokens = torch.tensor([3, 1], device="cuda", dtype=torch.int64)
    num_valid_tokens = None
    if layout == "normal":
        expert_tokens = None
        num_valid_tokens = torch.tensor([3], device="cuda", dtype=torch.int32)
        valid_rows = torch.arange(inputs.size(0), device="cuda") < num_valid_tokens
    else:
        rows_per_expert = inputs.size(0) // expert_tokens.numel()
        local_rows = torch.arange(rows_per_expert, device="cuda")
        valid_rows = (local_rows[None, :] < expert_tokens[:, None]).flatten()

    activation_impl = ACTIVATION_TYPE_IMPL_TEST_MAP[ActivationType(activation_type)]["impl"]
    options = dict(
        quant_mode=quant_mode.value,
        quant_dtype=quant_dtype,
        quant_group_size=128,
        group_scale_dtype=group_scale_dtype,
        activation_type=activation_type,
        layout=layout,
        expert_tokens=expert_tokens,
        num_valid_tokens=num_valid_tokens,
        zero_invalid=zero_invalid,
    )

    expected = process_input_ref(inputs, **options)
    actual = process_input(inputs, activation_impl=activation_impl, **options)

    assert_process_input_close(
        actual,
        expected,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        group_scale_dtype=group_scale_dtype,
        quant_group_size=128,
        valid_rows=valid_rows,
        zero_invalid=zero_invalid,
    )

import pytest
import torch

from humming import dtypes
from humming.config import InputQuantizationMode, ProcessInputProblemConfig
from humming.ops.input import process_input
from humming.testing.process_input import (
    assert_process_input_close,
    process_input_ref,
    skip_if_process_input_unsupported,
)
from humming.utils.math import ceil_div


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("quant_dtype", ["int4", "int8", "float8e4m3"])
@pytest.mark.parametrize(
    "shape_m,hidden_size,quant_group_size,quant_mode,group_scale_dtype,use_m_major_input_scale",
    [
        (3, 768, 128, "static_tensor", "float32", False),
        (129, 32768, 512, "dynamic_token", "float32", False),
        (2049, 768, 128, "dynamic_token", "float32", False),
        (129, 32768, 512, "dynamic_token", "float32", True),
        (3, 768, 128, "dynamic_token", "float32", True),
        (4, 7168, 128, "dynamic_group", "float32", False),
        (3, 768, 128, "dynamic_group", "float32", True),
        (3, 768, 128, "dynamic_group", "float8e4m3", False),
        (3, 768, 128, "dynamic_group", "float8e4m3", True),
        (3, 768, 128, "dynamic_group", "float8e8m0", False),
        (3, 768, 128, "dynamic_group", "float8e8m0", True),
        (3, 768, 128, "static_tensor_dynamic_group", "float32", False),
        (3, 768, 128, "static_tensor_dynamic_group", "float8e4m3", True),
        (3, 768, 128, "static_tensor_dynamic_group", "float8e8m0", False),
        (3, 768, 128, "dynamic_group_token", "float8e4m3", False),
        (129, 32768, 512, "dynamic_group_token", "float8e4m3", True),
        # Partial last group, e.g. a TP-split MoE intermediate (2304 / 4 = 576).
        (3, 576, 128, "dynamic_group", "float32", False),
        (3, 576, 128, "dynamic_group", "float32", True),
        (129, 288, 128, "dynamic_group", "float32", False),
        (2049, 2880, 128, "dynamic_group", "float32", False),
        (3, 576, 128, "dynamic_group", "float8e8m0", False),
        (3, 576, 128, "dynamic_group", "float8e4m3", True),
        (3, 576, 128, "static_tensor_dynamic_group", "float32", False),
        (3, 576, 128, "dynamic_group_token", "float8e4m3", False),
    ],
)
def test_quantization_scales(
    quant_dtype,
    shape_m,
    hidden_size,
    quant_group_size,
    quant_mode,
    group_scale_dtype,
    use_m_major_input_scale,
    use_pdl,
):
    skip_if_process_input_unsupported(quant_dtype, group_scale_dtype)
    torch.manual_seed(0)

    num_groups = ceil_div(hidden_size, quant_group_size)
    inputs = torch.randn(shape_m, num_groups, quant_group_size, device="cuda")

    group_exponents = torch.arange(num_groups, device="cuda") % 5 - 2
    group_amplitudes = torch.exp2(group_exponents.float())
    inputs *= group_amplitudes[None, :, None]
    inputs = inputs.flatten(1)[:, :hidden_size].contiguous()
    inputs[0] = 0

    quant_mode = InputQuantizationMode(quant_mode)
    static_tensor_scale = None
    if quant_mode.has_tensor_scale:
        static_tensor_scale = torch.tensor([0.5], device="cuda")

    options = dict(
        quant_mode=quant_mode.value,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        group_scale_dtype=group_scale_dtype,
        use_m_major_input_scale=use_m_major_input_scale,
    )

    expected = process_input_ref(inputs, static_tensor_scale=static_tensor_scale, **options)
    has_scale_tail = quant_mode.has_group_scale and group_scale_dtype != "float32" and num_groups % 4 != 0
    group_scales = torch.full_like(expected[1], -1) if has_scale_tail else None
    actual = process_input(
        inputs, group_scales=group_scales, token_scales=static_tensor_scale, use_pdl=use_pdl, **options
    )

    assert_process_input_close(
        actual,
        expected,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        group_scale_dtype=group_scale_dtype,
        quant_group_size=quant_group_size,
        use_m_major_input_scale=use_m_major_input_scale,
    )
    if has_scale_tail:
        scale_bytes = actual[1].view(torch.uint8)
        if use_m_major_input_scale:
            scale_tail = scale_bytes[-1].reshape(-1, 4)[:shape_m, num_groups % 4 :]
        else:
            scale_tail = scale_bytes[:, num_groups:]
        assert torch.count_nonzero(scale_tail) == 0


def test_group_token_scales_with_cold_inputs():
    """Launch on inputs no earlier launch has read, so warps reach the fused reductions at uneven times."""
    options = dict(
        quant_mode="dynamic_group_token",
        quant_dtype="int8",
        quant_group_size=128,
        group_scale_dtype="float8e4m3",
    )
    skip_if_process_input_unsupported(options["quant_dtype"], options["group_scale_dtype"])
    torch.manual_seed(0)

    # Each launch reads the first rows of its own 8 MiB region.
    shape_m, hidden_size, num_launches = 28, 1024, 16
    input_pool = torch.randn(num_launches, 2048, hidden_size, device="cuda")

    for launch_inputs in input_pool:
        inputs = launch_inputs[:shape_m]
        # The reference would warm the cache, so the kernel must read the slice first.
        actual = process_input(inputs, **options)
        expected = process_input_ref(inputs, **options)

        assert_process_input_close(actual, expected, **options)


@pytest.mark.parametrize("use_m_major_input_scale", [False, True])
@pytest.mark.parametrize(
    "hidden_size,quant_mode,quant_dtype,quant_group_size,group_scale_dtype",
    [
        (2880, "dynamic_group", "float8e4m3", 32, "float8e8m0"),
        (2880, "dynamic_group", "float4e2m1", 32, "float8e8m0"),
        (2880, "dynamic_group", "int8", 32, "float8e4m3"),
        (2848, "dynamic_group_token", "float4e2m1", 16, "float8e4m3"),
        (32800, "dynamic_group_token", "float4e2m1", 16, "float8e4m3"),
    ],
)
def test_packed_group_scale_padding_is_zeroed(
    hidden_size,
    quant_mode,
    quant_dtype,
    quant_group_size,
    group_scale_dtype,
    use_m_major_input_scale,
):
    """8-bit scales are packed four groups per word, and GEMMs with padded K read the unused groups of the
    last word. Start from a NaN-filled buffer so stale bytes cannot pass as zero."""
    skip_if_process_input_unsupported(quant_dtype, group_scale_dtype)
    torch.manual_seed(0)

    shape_m = 5
    num_groups = hidden_size // quant_group_size
    assert num_groups % 4 != 0
    inputs = torch.randn(shape_m, hidden_size, device="cuda")
    options = dict(
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        group_scale_dtype=group_scale_dtype,
        use_m_major_input_scale=use_m_major_input_scale,
    )

    problem = ProcessInputProblemConfig(input_dtype=dtypes.float32, hidden_size=hidden_size, **options)
    group_scales = torch.full(problem.get_group_scale_shape(shape_m), -1, dtype=torch.int32, device="cuda")
    actual = process_input(inputs, group_scales=group_scales, **options)
    expected = process_input_ref(inputs, **options)
    assert_process_input_close(actual, expected, **options)

    packed_rows = group_scales.T if use_m_major_input_scale else group_scales
    scale_bytes = packed_rows.contiguous().view(torch.uint8)[:shape_m]
    assert torch.all(scale_bytes[:, num_groups:] == 0)

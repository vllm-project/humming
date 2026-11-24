import torch
from torch._subclasses.fake_tensor import FakeTensor

from humming import dtypes
from humming.config import ActivationType, ProcessInputLayoutType, ProcessInputProblemConfig
from humming.kernel.process_input import ProcessInputKernel
from humming.ops.utils import init_humming_launcher, register_op
from humming.tune.process_input import get_process_input_tuning_intervals
from humming.utils.math import round_up

_process_input_cache: dict[tuple, tuple[ProcessInputProblemConfig, ProcessInputLayoutType, torch.Tensor]] = {}


def _make_process_input_problem_config(
    inputs: torch.Tensor,
    *,
    quant_mode,
    quant_dtype,
    quant_group_size,
    group_scale_key,
    activation_type,
    activation_impl,
    hadamard_block_size,
    layout,
    scatter_width,
    zero_invalid,
    use_m_major_input_scale,
) -> tuple[ProcessInputProblemConfig, ProcessInputLayoutType]:
    activation_type = ActivationType(activation_type)
    input_width = int(inputs.size(1))
    assert not activation_type.is_binary or input_width % 2 == 0

    hidden_size = input_width
    if activation_type.is_binary:
        hidden_size //= 2

    layout = ProcessInputLayoutType(layout)

    group_scale_dtype = group_scale_key
    if isinstance(group_scale_dtype, torch.dtype):
        group_scale_dtype = dtypes.DataType.from_torch_dtype(group_scale_dtype)

    config = ProcessInputProblemConfig(
        input_dtype=dtypes.DataType.from_torch_dtype(inputs.dtype),
        hidden_size=hidden_size,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        group_scale_dtype=group_scale_dtype,
        activation_type=activation_type,
        activation_impl=activation_impl,
        hadamard_block_size=hadamard_block_size,
        layout=layout,
        scatter_width=scatter_width,
        zero_invalid=zero_invalid,
        use_m_major_input_scale=use_m_major_input_scale,
    )
    return config, layout


@register_op("humming::prepare_process_input")
def _prepare_process_input_op(
    inputs: torch.Tensor,
    outputs: torch.Tensor | None = None,
    group_scales: torch.Tensor | None = None,
    token_scales: torch.Tensor | None = None,
    quant_mode: str = "none",
    quant_dtype: str | None = None,
    quant_group_size: int | None = None,
    group_scale_dtype: str | None = None,
    activation_type: str = "none",
    activation_impl: str | None = None,
    hadamard_block_size: int | None = None,
    layout: str = "normal",
    scatter_idx: torch.Tensor | None = None,
    zero_invalid: bool = False,
    use_m_major_input_scale: bool = False,
    use_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    assert inputs.ndim == 2, "process_input requires 2D inputs"
    is_scatter_layout = layout == ProcessInputLayoutType.Scatter
    if is_scatter_layout:
        assert scatter_idx is not None and scatter_idx.ndim == 2, "scatter requires 2D scatter_idx"
    scatter_width = int(scatter_idx.size(1)) if is_scatter_layout else 1
    group_scale_key = group_scale_dtype
    if group_scale_key is None and group_scales is not None and group_scales.dtype != torch.int32:
        group_scale_key = group_scales.dtype
    is_fake = isinstance(inputs, FakeTensor)
    # The problem config and kernel configs depend only on these values, not on tensor contents.
    cache_key = (
        inputs.get_device(),
        inputs.dtype,
        int(inputs.size(1)),
        quant_mode,
        quant_dtype,
        quant_group_size,
        group_scale_key,
        activation_type,
        activation_impl,
        hadamard_block_size,
        layout,
        scatter_width,
        zero_invalid,
        use_m_major_input_scale,
        use_pdl,
    )
    cached = None if is_fake else _process_input_cache.get(cache_key)
    if cached is not None:
        config, layout, configs = cached
    else:
        config, layout = _make_process_input_problem_config(
            inputs,
            quant_mode=quant_mode,
            quant_dtype=quant_dtype,
            quant_group_size=quant_group_size,
            group_scale_key=group_scale_key,
            activation_type=activation_type,
            activation_impl=activation_impl,
            hadamard_block_size=hadamard_block_size,
            layout=layout,
            scatter_width=scatter_width,
            zero_invalid=zero_invalid,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        configs = None

    num_input_rows = inputs.size(0)
    num_output_rows = num_input_rows
    if layout == ProcessInputLayoutType.Scatter:
        num_output_rows = num_input_rows * scatter_width
        if outputs is not None:
            num_output_rows = outputs.size(0)

    allocated_outputs = None
    if outputs is None:
        allocated_outputs = torch.empty(
            (num_output_rows, config.output_row_size),
            dtype=config.output_torch_dtype,
            device=inputs.device,
        )

    allocated_group_scales = None
    if config.quant_mode.has_group_scale and group_scales is None:
        allocated_group_scales = torch.empty(
            config.get_group_scale_shape(num_output_rows),
            dtype=config.group_scale_torch_dtype,
            device=inputs.device,
        )

    allocated_token_scales = None
    if config.quant_mode.has_token_scale and token_scales is None:
        token_scale_storage = torch.empty(
            round_up(num_output_rows, 4),
            dtype=torch.float32,
            device=inputs.device,
        )
        token_scale_shape = (num_output_rows, 1)
        if config.use_m_major_input_scale:
            token_scale_shape = (1, num_output_rows)

        allocated_token_scales = token_scale_storage[:num_output_rows].view(token_scale_shape)

    assert inputs.is_cuda
    if configs is None:
        with torch.cuda.device(inputs.device):
            if is_fake:
                init_humming_launcher()
                tuning_intervals = get_process_input_tuning_intervals(config, use_pdl)
                configs = torch.empty((len(tuning_intervals) * 4,), dtype=torch.int64, device="cpu")
            else:
                family_key = (inputs.device.index, config, use_pdl)
                configs = ProcessInputKernel._str2kernel_cache.get(family_key)
                if configs is None:
                    tuning_intervals = get_process_input_tuning_intervals(config, use_pdl)
                    configs = ProcessInputKernel.prepare_kernels(
                        config, tuning_intervals, inputs.device, family_key
                    )
                _process_input_cache[cache_key] = (config, layout, configs)

    return configs, allocated_outputs, allocated_group_scales, allocated_token_scales


def process_input(
    inputs: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    quant_mode: str = "none",
    quant_dtype: str | None = None,
    quant_group_size: int | None = None,
    group_scales: torch.Tensor | None = None,
    group_scale_dtype: str | None = None,
    token_scales: torch.Tensor | None = None,
    activation_type: str = "none",
    activation_impl: str | None = None,
    hadamard_block_size: int | None = None,
    layout: str = "normal",
    expert_tokens: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    num_valid_tokens: torch.Tensor | None = None,
    zero_invalid: bool = False,
    use_m_major_input_scale: bool = False,
    use_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Process inputs and return quantization scales.

    FP8 group scales store four consecutive groups per int32, low byte first.
    Their numerical format is selected by group_scale_dtype, not the tensor dtype.
    Scatter targets outside [0, scatter_idx.numel()) are ignored. Targets at or
    above num_valid_tokens are zeroed when zero_invalid is set, otherwise ignored.
    Normal layout applies the same valid-token bound to input row indices.
    Omitting num_valid_tokens makes all in-range targets valid.
    """
    options = dict(
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        group_scale_dtype=group_scale_dtype,
        activation_type=activation_type,
        activation_impl=activation_impl,
        hadamard_block_size=hadamard_block_size,
        layout=layout,
        scatter_idx=scatter_idx,
        zero_invalid=zero_invalid,
        use_m_major_input_scale=use_m_major_input_scale,
        use_pdl=use_pdl,
    )
    prepare_process_input = _prepare_process_input_op
    if torch.compiler.is_compiling() or isinstance(inputs, FakeTensor):
        prepare_process_input = torch.ops.humming.prepare_process_input

    prepared_tensors = prepare_process_input(inputs, outputs, group_scales, token_scales, **options)
    configs, allocated_outputs, allocated_group_scales, allocated_token_scales = prepared_tensors

    if outputs is inputs:
        torch.ops.humming.launch_process_input.inplace(
            configs, inputs, expert_tokens, scatter_idx, num_valid_tokens
        )
        return inputs, None, None

    if outputs is None:
        outputs = allocated_outputs
    if group_scales is None:
        group_scales = allocated_group_scales
    if token_scales is None:
        token_scales = allocated_token_scales

    assert outputs is not None
    torch.ops.humming.launch_process_input.default(
        configs,
        inputs,
        outputs,
        group_scales,
        token_scales,
        expert_tokens,
        scatter_idx,
        num_valid_tokens,
    )
    return outputs, group_scales, token_scales

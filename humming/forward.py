import json

import torch

from humming import dtypes, ops
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune import get_heuristics_class


def _resolve_use_pdl(
    config: LayerConfig,
    inputs: torch.Tensor,
    use_pdl: bool | None,
) -> bool:
    if use_pdl is not None:
        return use_pdl
    if not inputs.is_cuda:
        return False

    heuristics = get_heuristics_class(inputs.device)
    shape_m = inputs.numel() // inputs.size(-1)
    return heuristics.should_use_pdl_for_input(config, shape_m)


def _prepare_input_scale(config: LayerConfig, input_scale: torch.Tensor) -> torch.Tensor:
    mx_scale_dtype = str(config.as_dtype) in ("float8e4m3", "float8e8m0")
    grouped_mxmma = config.mma_type == MmaType.MXMMA and config.input_scale_group_size > 0
    if mx_scale_dtype and grouped_mxmma and input_scale.dtype != torch.int32:
        packed_scale = input_scale.view(torch.int32)
        if input_scale.ndim == 3:
            packed_scale = packed_scale.reshape(input_scale.size(0), input_scale.size(1))
        return packed_scale
    return input_scale


def _group_scale_layout(config: LayerConfig, m_major_scale: bool) -> str:
    if not m_major_scale or config.input_scale_group_size == 0:
        return "row_major"
    if str(config.as_dtype) in ("float8e4m3", "float8e8m0"):
        return "mx_packed"
    return "m_major"


def may_process_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    group_scales: torch.Tensor | None = None,
    token_scales: torch.Tensor | None = None,
    activation_type: str = "none",
    activation_impl: str | None = None,
    hadamard_block_size: int | None = None,
    layout: str = "normal",
    expert_layout: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    zero_invalid: bool = False,
    m_major_scale: bool = False,
    use_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    config.check_device(inputs.device)
    should_quantize = config.input_quant_mode.should_quantize
    quant_mode = "none"
    quant_dtype = None
    quant_group_size = None
    group_scale_dtype = None
    if should_quantize:
        assert config.as_dtype is not None
        quant_mode = config.input_quant_mode.value
        quant_dtype = str(config.a_dtype)
        quant_group_size = config.input_scale_group_size or None
        group_scale_dtype = str(config.as_dtype)

    no_transform = (
        activation_type == "none"
        and activation_impl in (None, "")
        and (hadamard_block_size is None or hadamard_block_size <= 1)
    )
    no_layout = layout == "normal" and expert_layout is None and indices is None and not zero_invalid
    no_buffers = outputs is None and group_scales is None and token_scales is None
    if not should_quantize and no_transform and no_layout and no_buffers:
        return inputs, None, None

    resolved_use_pdl = _resolve_use_pdl(config, inputs, use_pdl)
    return ops.process_input(
        inputs=inputs,
        outputs=outputs,
        quant_mode=quant_mode,
        quant_dtype=quant_dtype,
        quant_group_size=quant_group_size,
        group_scales=group_scales,
        group_scale_dtype=group_scale_dtype,
        token_scales=token_scales,
        activation_type=activation_type,
        activation_impl=activation_impl,
        hadamard_block_size=hadamard_block_size,
        layout=layout,
        expert_layout=expert_layout,
        indices=indices,
        zero_invalid=zero_invalid,
        group_scale_layout=_group_scale_layout(config, m_major_scale),
        use_pdl=resolved_use_pdl,
    )


def may_quant_input(
    config: LayerConfig,
    inputs: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    quanted_input: torch.Tensor | None = None,
    use_pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if config.a_dtype.num_bits == 16:
        config.check_device(inputs.device)
        return inputs, None
    if input_scale is not None:
        config.check_device(inputs.device)
        return inputs, input_scale
    outputs, group_scales, token_scales = may_process_input(
        config,
        inputs=inputs,
        outputs=quanted_input,
        m_major_scale=(config.mma_type == MmaType.MXMMA and config.input_scale_group_size > 0),
        use_pdl=use_pdl,
    )
    if token_scales is not None:
        token_scales = token_scales.unsqueeze(-1)
    scale = group_scales if group_scales is not None else token_scales
    assert scale is not None
    return outputs, _prepare_input_scale(config, scale)


def humming_forward(
    config: LayerConfig,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    weight_scale_2: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    input_scale_2: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
    compute_config: dict | str | None = None,
    tuning_config: dict | list | str | None = None,
    hadamard_block_size: int | None = None,
    use_pdl: bool | None = None,
) -> torch.Tensor:
    parsed_compute_config = compute_config
    if isinstance(parsed_compute_config, str) and parsed_compute_config:
        parsed_compute_config = json.loads(parsed_compute_config)

    m_major_scale = False
    if config.input_scale_group_size > 0:
        if isinstance(parsed_compute_config, dict):
            m_major_scale = bool(parsed_compute_config.get("use_m_major_input_scale", False))

    gemm_type = None
    if isinstance(parsed_compute_config, dict):
        gemm_type_value = parsed_compute_config.get("gemm_type")
        if gemm_type_value is not None:
            gemm_type = GemmType(gemm_type_value)

    inputs_are_quantized = False
    if config.input_quant_mode.should_quantize:
        quantized_torch_dtype = dtypes.torch_dtype_map.get(config.a_dtype, torch.uint8)
        inputs_are_quantized = inputs.dtype == quantized_torch_dtype
        if config.a_dtype.num_bits == 4:
            inputs_are_quantized = inputs.dtype == torch.uint8

    needs_transform = hadamard_block_size is not None and hadamard_block_size > 1
    should_process = False
    if not inputs_are_quantized:
        should_process = config.input_quant_mode.should_quantize
        if needs_transform:
            should_process = True
    if should_process:
        group_scales = input_scale if config.input_quant_mode.uses_group_scale else None
        token_scales = input_scale_2 if config.input_quant_mode.has_secondary_scale else input_scale

        process_inputs = inputs
        process_layout = "normal"
        process_expert_layout = None
        flatten_grouped_padded = False
        if config.input_quant_mode.has_static_tensor_scale and config.num_experts > 0:
            if gemm_type == GemmType.GROUPED_CONTIGUOUS:
                process_layout = "grouped"
                process_expert_layout = expert_layout
            elif gemm_type == GemmType.GROUPED_MASKED:
                assert expert_layout is not None, "grouped_masked input processing requires expert_layout"
                assert inputs.ndim == 2 and inputs.size(0) % config.num_experts == 0
                process_inputs = inputs.view(config.num_experts, -1, inputs.size(-1))
                process_layout = "grouped_padded"
                process_expert_layout = expert_layout
                flatten_grouped_padded = True
            elif gemm_type == GemmType.INDEXED and config.num_experts > 1:
                raise ValueError(
                    "indexed GEMM cannot use per-expert static input scales because "
                    "its quantized inputs are shared across experts"
                )

        inputs, group_scales, token_scales = may_process_input(
            config,
            inputs=process_inputs,
            group_scales=group_scales,
            token_scales=token_scales,
            hadamard_block_size=hadamard_block_size,
            layout=process_layout,
            expert_layout=process_expert_layout,
            m_major_scale=m_major_scale,
            use_pdl=use_pdl,
        )
        if flatten_grouped_padded:
            inputs = inputs.view(-1, inputs.size(-1))
            if group_scales is not None and not m_major_scale:
                group_scales = group_scales.view(-1, group_scales.size(-1))
            if token_scales is not None and config.input_quant_mode.has_dynamic_token_scale:
                token_scales = token_scales.reshape(-1)
        if token_scales is not None and config.input_quant_mode.has_dynamic_token_scale:
            token_scales = token_scales.unsqueeze(-1)
        input_scale = group_scales if config.input_quant_mode.uses_group_scale else token_scales
        input_scale_2 = token_scales if config.input_quant_mode.has_secondary_scale else None
        if input_scale is not None:
            input_scale = _prepare_input_scale(config, input_scale)

    if isinstance(compute_config, dict):
        compute_config = json.dumps(compute_config)
    if isinstance(tuning_config, (list, dict)):
        tuning_config = json.dumps(tuning_config)

    return ops.humming_gemm(
        layer_config=config.to_str(),
        compute_config=compute_config,
        tuning_config=tuning_config,
        inputs=inputs,
        weight=weight,
        outputs=outputs,
        input_scale=input_scale,
        input_scale_2=input_scale_2,
        weight_scale=weight_scale,
        zero_point=zero_point,
        bias=bias,
        weight_scale_2=weight_scale_2,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        locks=locks,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
    )

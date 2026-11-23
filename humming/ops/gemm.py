import json

import torch
from torch._subclasses.fake_tensor import FakeTensor

from humming import dtypes
from humming.config import LayerConfig, ProcessInputProblemConfig
from humming.kernel.humming import HummingKernel
from humming.ops.launcher import launch_kernel
from humming.ops.utils import _should_use_torch_op, init_humming_launcher, register_op


@torch.compiler.assume_constant_result
def _get_humming_gemm_output_meta(layer_config: str, compute_config: str | None):
    layer_obj = json.loads(layer_config)
    compute_obj = json.loads(compute_config or "{}")
    shape_n = int(layer_obj["shape_n"]) - int(layer_obj.get("pad_shape_n", 0))
    c_dtype = dtypes.DataType.from_str(layer_obj["c_dtype"])
    divisor = 2 if layer_obj.get("activation_func_type") in ("binary_split", "binary_interleaved") else 1
    shape_n //= divisor
    if layer_obj.get("requantize_output", False):
        c_dtype = dtypes.DataType.from_str(layer_obj["a_dtype"])
        shape_n = shape_n * min(8, c_dtype.num_bits) // 8
    output_dtype = dtypes.torch_dtype_map.get(c_dtype, torch.uint8)
    return shape_n, output_dtype, compute_obj.get("gemm_type") == "indexed"


@torch.compiler.assume_constant_result
def _get_fused_output_scale_meta(
    layer_config: str, compute_config: str | None
) -> tuple[int, torch.dtype, bool] | None:
    layer_obj = json.loads(layer_config)
    if not layer_obj.get("requantize_output", False):
        return None

    config = LayerConfig(**layer_obj)
    output_config = ProcessInputProblemConfig(
        input_dtype=config.c_dtype,
        hidden_size=config.output_shape_n,
        quant_mode=config.input_quant_mode,
        quant_dtype=config.a_dtype,
        quant_group_size=config.input_scale_group_size,
        group_scale_dtype=config.as_dtype,
    )
    compute_obj = json.loads(compute_config or "{}")
    m_major_scale = bool(compute_obj.get("use_m_major_input_scale", False))
    return output_config.get_group_scale_shape(1)[1], output_config.group_scale_torch_dtype, m_major_scale


@register_op("humming::prepare_kernels")
def _prepare_kernels_op(
    device_guard: torch.Tensor,
    layer_config: str,
    compute_config: str | None = None,
    tuning_config: str | None = None,
) -> torch.Tensor:
    device = device_guard.device
    if isinstance(device_guard, FakeTensor):
        with torch.cuda.device(device):
            init_humming_launcher()
            _, _, tuning_obj = HummingKernel._resolve_configs(
                layer_config,
                compute_config,
                tuning_config,
                device,
            )
        num_configs = len(tuning_obj) if isinstance(tuning_obj, list) else 1
        return torch.empty((num_configs * 4,), dtype=torch.int64, device="cpu")

    return HummingKernel.prepare_kernels(layer_config, compute_config, tuning_config, device)


def humming_gemm(
    *,
    layer_config: str,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None = None,
    compute_config: str | None = None,
    tuning_config: str | None = None,
    input_scale: torch.Tensor | None = None,
    input_scale_2: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    outputs: torch.Tensor | None = None,
    output_scales: torch.Tensor | None = None,
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    scale_meta = _get_fused_output_scale_meta(layer_config, compute_config)
    if scale_meta is not None and output_scales is None:
        scale_columns, scale_dtype, m_major_scale = scale_meta
        _, _, is_indexed = _get_humming_gemm_output_meta(layer_config, compute_config)
        rows = inputs.size(0) * (top_k if is_indexed else 1)
        scale_shape = (scale_columns, (rows + 3) // 4 * 4) if m_major_scale else (rows, scale_columns)
        output_scales = torch.empty(scale_shape, dtype=scale_dtype, device=inputs.device)

    use_ops = _should_use_torch_op(inputs)
    if use_ops:
        if outputs is None:
            shape_n, output_dtype, is_indexed = _get_humming_gemm_output_meta(layer_config, compute_config)
            shape_m = inputs.size(0) * (top_k if is_indexed else 1)
            outputs = inputs.new_empty((shape_m, shape_n), dtype=output_dtype)
        configs = torch.ops.humming.prepare_kernels(inputs, layer_config, compute_config, tuning_config)
    else:
        configs = _prepare_kernels_op(inputs, layer_config, compute_config, tuning_config)

    result = launch_kernel(
        configs=configs,
        inputs=inputs,
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        input_scale=input_scale,
        input_scale_2=input_scale_2,
        zero_point=zero_point,
        bias=bias,
        outputs=outputs,
        output_scales=output_scales,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        locks=locks,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
    )

    if scale_meta is not None:
        assert output_scales is not None
        return result, output_scales
    return result

import json

import torch
from torch._subclasses.fake_tensor import FakeTensor

from humming import dtypes
from humming.kernel.humming import HummingKernel
from humming.ops.launcher import launch_kernel
from humming.ops.utils import _should_use_torch_op, init_humming_launcher, register_op
from humming.utils.rocm import is_rocm_build


@torch.compiler.assume_constant_result
def _get_humming_gemm_output_meta(layer_config: str, compute_config: str | None):
    layer_obj = json.loads(layer_config)
    compute_obj = json.loads(compute_config or "{}")
    shape_n = int(layer_obj["shape_n"]) - int(layer_obj.get("pad_shape_n", 0))
    c_dtype = dtypes.DataType.from_str(layer_obj["c_dtype"])
    output_dtype = dtypes.torch_dtype_map[c_dtype]
    return shape_n, output_dtype, compute_obj.get("gemm_type") == "indexed"


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
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    """Run GEMM with FP8 input group scales packed four per int32.

    The layer configuration determines the numerical scale format. Packed scales
    have shape (rows, packed_groups), or (packed_groups, padded_rows) for m-major.
    """
    if is_rocm_build():
        return _humming_gemm_rocm(
            layer_config=layer_config,
            inputs=inputs,
            weight=weight,
            weight_scale=weight_scale,
            weight_scale_2=weight_scale_2,
            compute_config=compute_config,
            tuning_config=tuning_config,
            input_scale=input_scale,
            input_scale_2=input_scale_2,
            zero_point=zero_point,
            bias=bias,
            outputs=outputs,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            expert_layout=expert_layout,
            locks=locks,
            top_k=top_k,
            valid_shape_m=valid_shape_m,
        )
    return _humming_gemm(
        layer_config=layer_config,
        inputs=inputs,
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        compute_config=compute_config,
        tuning_config=tuning_config,
        input_scale=input_scale,
        input_scale_2=input_scale_2,
        zero_point=zero_point,
        bias=bias,
        outputs=outputs,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        locks=locks,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
    )


def _humming_gemm_rocm(
    *, layer_config: str, inputs: torch.Tensor, outputs: torch.Tensor | None, bias, **kwargs
):
    """ROCm MFMA GEMM: pads K / N for the kernel's 128 / 64 alignment and adds the bias here."""
    from humming.kernel.humming_mfma import get_mfma_gemm_meta, get_mfma_num_experts

    shape_n, pad_shape_n, pad_shape_k = get_mfma_gemm_meta(layer_config)
    if pad_shape_k:
        inputs = _pad_rocm_inputs(inputs, pad_shape_k)
    if get_mfma_num_experts(layer_config):
        # MoE: the kernel writes only the unpadded N columns and adds the per-expert bias itself.
        if outputs is None:
            out_n, output_dtype, is_indexed = _get_humming_gemm_output_meta(
                layer_config, kwargs.get("compute_config")
            )
            rows = inputs.size(0) * (kwargs.get("top_k", 1) if is_indexed else 1)
            outputs = inputs.new_empty((rows, out_n), dtype=output_dtype)
        return _humming_gemm(layer_config=layer_config, inputs=inputs, outputs=outputs, bias=bias, **kwargs)
    if pad_shape_n:
        _, output_dtype, _ = _get_humming_gemm_output_meta(layer_config, kwargs.get("compute_config"))
        padded_outputs = inputs.new_empty((inputs.size(0), shape_n), dtype=output_dtype)
        padded_outputs = _humming_gemm(
            layer_config=layer_config, inputs=inputs, outputs=padded_outputs, bias=None, **kwargs
        )
        result = padded_outputs[:, : shape_n - pad_shape_n]
        if bias is not None:
            result = result + bias[: result.size(1)]
        if outputs is None:
            return result.contiguous()
        outputs.copy_(result)
        return outputs
    result = _humming_gemm(layer_config=layer_config, inputs=inputs, outputs=outputs, bias=None, **kwargs)
    if bias is not None:
        result.add_(bias)
    return result


def _pad_rocm_inputs(inputs: torch.Tensor, pad_shape_k: int) -> torch.Tensor:
    """Zero-pad K. FP8 / INT8 / packed FP4 (uint8, two values per byte) inputs are padded as bytes."""
    if inputs.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.nn.functional.pad(inputs, (0, pad_shape_k))
    pad_bytes = pad_shape_k // 2 if inputs.dtype == torch.uint8 else pad_shape_k
    padded = torch.nn.functional.pad(inputs.view(torch.uint8), (0, pad_bytes))
    return padded.view(inputs.dtype)


def _humming_gemm(
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
    sorted_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    expert_layout: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
    top_k: int = 1,
    valid_shape_m: int = 0,
) -> torch.Tensor:
    use_ops = _should_use_torch_op(inputs)
    if use_ops:
        if outputs is None:
            shape_n, output_dtype, is_indexed = _get_humming_gemm_output_meta(layer_config, compute_config)
            shape_m = inputs.size(0) * (top_k if is_indexed else 1)
            outputs = inputs.new_empty((shape_m, shape_n), dtype=output_dtype)
        configs = torch.ops.humming.prepare_kernels(inputs, layer_config, compute_config, tuning_config)
    else:
        configs = _prepare_kernels_op(inputs, layer_config, compute_config, tuning_config)

    return launch_kernel(
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
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        expert_layout=expert_layout,
        locks=locks,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
    )

import json
from dataclasses import replace
from functools import partial

import pytest
import torch

from humming import dtypes, ops
from humming.config import ActivationType, ComputeConfig, GemmType, LayerConfig, ProcessInputProblemConfig
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.kernel.humming import HummingKernel
from humming.ops import gemm
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)
from humming.testing.data import generate_random_tensor, generate_random_topk_ids
from humming.testing.process_input import ACTIVATION_TYPE_IMPL_TEST_MAP, apply_activation_ref
from humming.transform import prepare_and_transform_humming_tensors
from humming.tune import get_heuristics_config


def _case(name, a_dtype, activation, gemm_type="dense", *, requantize=False, m_major=False, pad_shape_n=64):
    activation = ActivationType(activation)
    gemm_type = GemmType(gemm_type)
    quantized_inputs = a_dtype.num_bits < 16
    group_size = 16 if a_dtype.num_bits == 4 else 32
    activation_impl = ACTIVATION_TYPE_IMPL_TEST_MAP[activation]["impl"] or None
    activation_ref = partial(apply_activation_ref, activation=activation)
    if activation == ActivationType.Unary:
        activation_impl = "__fdividef(a, 1.f + __expf(-a))"
        activation_ref = torch.nn.functional.silu
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=256,
            pad_shape_n=pad_shape_n,
            shape_k=1024,
            num_experts=0 if gemm_type == GemmType.DENSE else 4,
            a_dtype=a_dtype,
            b_dtype=dtypes.float4e2m1 if quantized_inputs else dtypes.uint4,
            c_dtype=dtypes.bfloat16 if quantized_inputs else a_dtype,
            bs_dtype=(dtypes.float8e4m3 if group_size == 16 else dtypes.float8e8m0)
            if quantized_inputs
            else a_dtype,
            input_scale_group_size=group_size if quantized_inputs else 0,
            weight_scale_group_size=group_size if quantized_inputs else 128,
            has_bias=True,
            has_zero_point=not quantized_inputs,
            activation_func_type=activation,
            activation_func_impl=activation_impl,
            requantize_output=requantize,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type, use_m_major_input_scale=m_major),
        top_k=1 if gemm_type == GemmType.DENSE else 2,
        seed=2026,
        output_activation_ref=activation_ref,
    )


FUSED_OUTPUT_CASES = (
    _case("bf16-unary", dtypes.bfloat16, "unary"),
    _case("fp8-unary", dtypes.float8e4m3, "unary"),
    _case("fp8-unary-partial-n", dtypes.float8e4m3, "unary", pad_shape_n=32),
    _case("fp8-unary-indexed", dtypes.float8e4m3, "unary", "indexed"),
    _case("fp8-unary-grouped", dtypes.float8e4m3, "unary", "grouped_contiguous"),
    _case("fp8-unary-masked", dtypes.float8e4m3, "unary", "grouped_masked"),
    _case("fp16-glu", dtypes.float16, "binary_split"),
    _case("fp8-activation-indexed", dtypes.float8e4m3, "binary_interleaved", "indexed"),
    _case("fp8-requant", dtypes.float8e4m3, "none", requantize=True),
    _case("fp8-glu-indexed", dtypes.float8e4m3, "binary_split", "indexed", requantize=True),
    _case(
        "fp8-glu-grouped",
        dtypes.float8e4m3,
        "binary_split",
        "grouped_contiguous",
        requantize=True,
        m_major=True,
    ),
    _case("fp4-glu-masked", dtypes.float4e2m1, "binary_interleaved", "grouped_masked", requantize=True),
)


@pytest.mark.parametrize("test_case", FUSED_OUTPUT_CASES, ids=str)
def test_fused_output(test_case):
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


@pytest.mark.parametrize("test_case", FUSED_OUTPUT_CASES, ids=str)
def test_fused_output_rejects_stream_k(test_case):
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
    compute = test_case.compute_config
    tuning = get_heuristics_config(config, shape_m=128, **compute.to_dict())
    assert tuning["use_stream_k"] is False
    intervals = get_heuristics_config(config, **compute.to_dict())
    assert all(entry[2]["use_stream_k"] is False for entry in intervals)
    with pytest.raises(AssertionError, match="fused output requires use_stream_k=False"):
        HummingKernel.prepare_kernels(config.to_str(), compute.to_str(), dict(tuning, use_stream_k=True))


@pytest.mark.parametrize("requantize", [False, True], ids=["swiglu", "fp8-requant"])
def test_grouped_pdl_reuses_output_descriptors(requantize):
    if current_device.sm_version != 120:
        pytest.skip("Regression uses SM120 MMA and MXMMA launch geometries")
    config = LayerConfig(
        shape_n=1024,
        shape_k=4096,
        num_experts=8,
        a_dtype=dtypes.float8e4m3 if requantize else dtypes.bfloat16,
        b_dtype=dtypes.float4e2m1 if requantize else dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0 if requantize else dtypes.bfloat16,
        input_scale_group_size=32 if requantize else 0,
        weight_scale_group_size=32 if requantize else 128,
        has_bias=True,
    )
    compute = ComputeConfig(gemm_type=GemmType.GROUPED_CONTIGUOUS)
    runner = KernelTestRunner(
        KernelTestCase(
            name="grouped-pdl-descriptors", layer_config=config, compute_config=compute, top_k=2, seed=219
        )
    )
    config = runner.layer_config
    shape_m = 4096
    torch.manual_seed(219 + shape_m)
    inputs = generate_random_tensor((shape_m, config.shape_k), dtype=config.param_dtype, device="cuda")
    topk_ids = generate_random_topk_ids(shape_m, config.num_experts, 2, device="cuda")
    block_m = 96 if requantize else 128
    routed, routing, _ = runner._prepare_problem(shape_m, inputs, topk_ids, block_m)
    _, inputs, input_scale, input_scale_2 = runner.prepare_inputs(routed)
    outputs = torch.empty((routed.shape[0], config.shape_n), dtype=config.param_dtype, device="cuda")
    tuning = dict(use_pdl=True, use_tma=True, use_warp_spec=True, evict_weights_first=False)
    if requantize:
        tuning.update(mma_type="mxmma", num_stages=2, smem_reuse_mode="last_stage", use_stream_k=False)
        plain_tuning = dict(tuning, block_shape=[96, 256, 128], warp_shape=[96, 32, 128])
        fused_tuning = dict(tuning, block_shape=[96, 256, 64], warp_shape=[96, 32, 64])
    else:
        tuning.update(mma_type="mma")
        plain_tuning = dict(
            tuning, block_shape=[128, 256, 64], warp_shape=[128, 32, 64], num_stages=3, use_stream_k=True
        )
        fused_tuning = dict(
            tuning, block_shape=[128, 128, 64], warp_shape=[128, 32, 32], num_stages=4, use_stream_k=False
        )
    fused_config = replace(
        config,
        activation_func_type="none" if requantize else "binary_interleaved",
        activation_func_impl=None if requantize else "__fdividef(a, 1.f + __expf(-a)) * b",
        requantize_output=requantize,
    )
    kwargs = dict(
        inputs=inputs,
        input_scale=input_scale,
        input_scale_2=input_scale_2,
        compute_config=compute.to_str(),
        top_k=2,
        valid_shape_m=shape_m * 2,
        locks=torch.zeros(2048, dtype=torch.int32, device="cuda"),
        **routing,
        **runner.kernel_tensors,
    )

    def plain():
        return ops.humming_gemm(
            layer_config=config.to_str(), tuning_config=json.dumps(plain_tuning), outputs=outputs, **kwargs
        )

    # Allocate persistent outputs before testing allocator reuse of the launcher's temporary descriptors.
    plain_reference = plain().clone()
    fused_result = ops.humming_gemm(
        layer_config=fused_config.to_str(), tuning_config=json.dumps(fused_tuning), **kwargs
    )
    fused_outputs, fused_scales = fused_result if requantize else (fused_result, None)
    fused_reference = fused_outputs.clone()
    scale_reference = fused_scales.clone() if requantize else None
    torch.cuda.synchronize()

    def fused():
        return ops.humming_gemm(
            layer_config=fused_config.to_str(),
            tuning_config=json.dumps(fused_tuning),
            outputs=fused_outputs,
            output_scales=fused_scales,
            **kwargs,
        )

    # Keep launches asynchronous: synchronization between kernels masks the descriptor race.
    for _ in range(100):
        fused()
        plain()
    torch.cuda.synchronize()
    runner._assert_close(outputs, plain_reference, plain_tuning["use_stream_k"])
    torch.testing.assert_close(fused_outputs.float(), fused_reference.float(), rtol=0, atol=0)
    if requantize:
        torch.testing.assert_close(fused_scales, scale_reference, rtol=0, atol=0)


@pytest.mark.parametrize("test_case", FUSED_OUTPUT_CASES, ids=str)
@pytest.mark.parametrize("provide_scales", [False, True])
@pytest.mark.parametrize("compile_forward", [False, True])
def test_gemm_output_scales(monkeypatch, test_case, provide_scales, compile_forward):
    config = test_case.layer_config
    shape_m = 17
    output_rows = shape_m * (test_case.top_k if test_case.compute_config.gemm_type == GemmType.INDEXED else 1)
    expected_scales = None
    if config.requantize_output:
        quant_config = ProcessInputProblemConfig(
            input_dtype=config.c_dtype,
            hidden_size=config.output_shape_n,
            quant_mode=config.input_quant_mode,
            quant_dtype=config.a_dtype,
            quant_group_size=config.input_scale_group_size,
            group_scale_dtype=config.as_dtype,
            use_m_major_input_scale=test_case.compute_config.use_m_major_input_scale,
        )
        expected_scales = torch.empty(
            quant_config.get_group_scale_shape(output_rows), dtype=quant_config.group_scale_torch_dtype
        )
    supplied_scales = expected_scales if provide_scales else None
    outputs = torch.empty((output_rows, 1))

    def launch_kernel(**kwargs):
        if kwargs["output_scales"] is not None:
            kwargs["output_scales"].fill_(1)
        return kwargs["outputs"]

    # Isolate the Python allocation and return contract from CUDA kernel execution.
    monkeypatch.setattr(gemm, "_should_use_torch_op", lambda inputs: False)
    monkeypatch.setattr(gemm, "_prepare_kernels_op", lambda *args: None)
    monkeypatch.setattr(gemm, "launch_kernel", launch_kernel)

    layer_config = config.to_str()
    compute_config = test_case.compute_config.to_str()

    def forward(inputs):
        return gemm.humming_gemm(
            layer_config=layer_config,
            compute_config=compute_config,
            inputs=inputs,
            weight=inputs,
            weight_scale=None,
            outputs=outputs,
            output_scales=supplied_scales,
            top_k=test_case.top_k,
        )

    if compile_forward:
        torch._dynamo.reset()
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    result = forward(torch.empty((shape_m, config.shape_k)))
    if config.requantize_output:
        result_outputs, result_scales = result
        assert result_outputs is outputs
        assert result_scales.shape == expected_scales.shape
        assert result_scales.dtype == expected_scales.dtype
        assert result_scales.device == outputs.device
        torch.testing.assert_close(result_scales, torch.ones_like(expected_scales))
        if provide_scales:
            assert result_scales is supplied_scales
    else:
        assert result is outputs


@pytest.mark.parametrize("requantize_output", [False, True])
def test_prepare_glu_layouts(requantize_output):
    input_dtype = dtypes.float8e4m3 if requantize_output else dtypes.bfloat16
    skip_if_unsupported(a_dtype=input_dtype)
    weight_schema = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1 if requantize_output else dtypes.uint4,
        bs_dtype=dtypes.float8e8m0 if requantize_output else dtypes.bfloat16,
        weight_scale_group_size=32 if requantize_output else 128,
        has_zero_point=not requantize_output,
    )
    input_schema = HummingInputSchema(
        a_dtype=input_dtype,
        input_scale_group_size=32 if requantize_output else 0,
        input_scale_dtype=dtypes.float8e8m0 if requantize_output else None,
    )
    torch.manual_seed(2026)
    shape_n, shape_k = 192, 1024
    weight = torch.randn((shape_n, shape_k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((shape_n,), dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(shape_n, device="cuda").reshape(2, -1).T.flatten()
    transformed_layouts = []
    for activation in ("binary_split", "binary_interleaved"):
        source_weight = weight if activation == "binary_split" else weight[indices]
        tensors = weight_schema.quant_tensor(source_weight, weight_schema, torch.bfloat16)
        tensors["bias"] = bias if activation == "binary_split" else bias[indices]
        config, transformed = prepare_and_transform_humming_tensors(
            tensors,
            shape_n=shape_n,
            shape_k=shape_k,
            weight_schema=weight_schema,
            input_schema=input_schema,
            pad_n_to_multiple=256,
            has_bias=True,
            torch_dtype=torch.bfloat16,
            device=torch.device("cuda"),
            activation_func_type=activation,
            activation_func_impl="a * b",
            requantize_output=requantize_output,
        )
        assert config.activation_func_type == ActivationType(activation)
        assert config.activation_func_impl == "a * b"
        assert config.requantize_output == requantize_output
        transformed_layouts.append(transformed)

    assert transformed_layouts[0].keys() == transformed_layouts[1].keys()
    for name, tensor in transformed_layouts[0].items():
        torch.testing.assert_close(tensor, transformed_layouts[1][name], rtol=0, atol=0)

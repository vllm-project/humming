from functools import partial

import pytest
import torch

from humming import dtypes
from humming.config import ActivationType, ComputeConfig, GemmType, LayerConfig, ProcessInputProblemConfig
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.ops import gemm
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)
from humming.testing.process_input import ACTIVATION_TYPE_IMPL_TEST_MAP, apply_activation_ref
from humming.transform import prepare_and_transform_humming_tensors


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

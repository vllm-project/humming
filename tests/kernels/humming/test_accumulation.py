import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.config.mma import get_default_mma_type
from humming.layer import HummingLayer
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)


def _case(
    name: str,
    *,
    a_dtype,
    b_dtype,
    shape_k: int = 1024,
    input_scale_group_size: int = 0,
    weight_scale_group_size: int = 0,
    bs_dtype=dtypes.float16,
    has_bias: bool = False,
    pad_shape_n: int = 0,
    pad_shape_k: int = 0,
    gemm_type: GemmType = GemmType.DENSE,
    num_experts: int = 0,
) -> KernelTestCase:
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=1024,
            shape_k=shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            num_experts=num_experts,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=dtypes.float16,
            bs_dtype=bs_dtype,
            input_scale_group_size=input_scale_group_size,
            weight_scale_group_size=weight_scale_group_size,
            has_bias=has_bias,
        ),
        compute_config=ComputeConfig(
            gemm_type=gemm_type,
            use_f16_accum=True,
        ),
        top_k=2 if gemm_type != GemmType.DENSE else 1,
        seed=2026,
    )


F16_ACCUM_CASES = (
    _case(
        "fp16-channel-scale",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
    ),
    _case(
        "fp16-group-scale",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
        weight_scale_group_size=64,
    ),
    _case(
        "fp16-group-scale-bias",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
        weight_scale_group_size=64,
        has_bias=True,
    ),
    _case(
        "fp16-bias-pad-nk",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
        has_bias=True,
        pad_shape_n=24,
        pad_shape_k=32,
    ),
    _case(
        "fp8-integer-weight",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
    ),
    _case(
        "fp8-fp4-weight",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
    ),
    _case(
        "fp8-grouped-input-and-weight",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint4,
        input_scale_group_size=64,
        weight_scale_group_size=64,
    ),
    _case(
        "fp8-f4-grouped-input-and-weight-32",
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        input_scale_group_size=32,
        weight_scale_group_size=32,
    ),
    _case(
        "fp16-indexed-bias",
        a_dtype=dtypes.float16,
        b_dtype=dtypes.uint4,
        has_bias=True,
        gemm_type=GemmType.GROUPED_CONTIGUOUS,
        num_experts=8,
    ),
)


@pytest.mark.parametrize("test_case", F16_ACCUM_CASES, ids=str)
def test_f16_accumulation(test_case):
    config = test_case.layer_config
    assert test_case.compute_config.use_f16_accum
    assert config.c_dtype == dtypes.float16
    if config.use_block_scaled_mma:
        pytest.skip("block-scaled layers require FP32 accumulation")
    if config.use_raw_weight and config.a_dtype.num_bits != config.b_dtype.num_bits:
        pytest.skip("native mixed weight layout requires UMMA with FP32 accumulation")
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)

    runner = KernelTestRunner(test_case)
    results = runner.run()
    assert_kernel_test_shape_coverage(results)
    assert all(result.tuning_values["use_f16_accum"] for result in results)


def test_f16_accumulation_case_coverage():
    assert {case.layer_config.a_dtype for case in F16_ACCUM_CASES} == {dtypes.float16, dtypes.float8e4m3}
    assert any(case.layer_config.has_bias for case in F16_ACCUM_CASES)
    assert any(case.layer_config.pad_shape_k for case in F16_ACCUM_CASES)
    assert any(case.layer_config.weight_scale_group_size for case in F16_ACCUM_CASES)
    assert any(
        case.layer_config.weight_scale_group_size and case.layer_config.has_bias for case in F16_ACCUM_CASES
    )
    assert any(case.layer_config.input_scale_group_size for case in F16_ACCUM_CASES)
    assert not any(case.layer_config.use_fused_e8m0_scale for case in F16_ACCUM_CASES)
    assert any(case.compute_config.gemm_type == GemmType.GROUPED_CONTIGUOUS for case in F16_ACCUM_CASES)


def _make_dense_stream_k_layer(shape_n, *, nonzero=False, bias=0.0, scaled=False):
    shape_k = 2048
    weights = torch.full((shape_n, shape_k), 64.0, dtype=torch.float32)
    weights[:, shape_k // 2 :] *= -1
    if nonzero:
        weights[:, shape_k // 2] = 0
    scales = torch.ones((shape_n // 128, shape_k // 128), dtype=torch.float32)
    if scaled:
        scales *= (2.0 ** (torch.arange(shape_n // 128) % 4 - 1)).reshape(-1, 1)

    with torch.device("cuda"):
        layer = HummingLayer(
            shape_n=shape_n,
            shape_k=shape_k,
            has_bias=bias != 0,
            torch_dtype=torch.float16,
            weight_config=HummingWeightSchema(
                b_dtype="float8e4m3", weight_scale_group_size=128, weight_scale_group_size_n=128
            ),
            input_config=HummingInputSchema(),
        )
    layer.weight.data.copy_(weights.to(torch.float8_e4m3fn).view(torch.int32))
    layer.weight_scale.data.copy_(scales)
    if bias:
        layer.bias.data.fill_(bias)
    layer.transform()
    reference_weights = weights * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return layer, reference_weights


def _dense_stream_k_tuning(mma_type):
    skip_if_unsupported(a_dtype=dtypes.float16, b_dtype=dtypes.float8e4m3, mma_type=mma_type)
    use_wgmma = mma_type == "wgmma"
    return dict(
        mma_type=mma_type,
        block_shape=(48, 128, 64 if use_wgmma else 128),
        warp_shape=(48, 32 if use_wgmma else 64, 32),
        num_stages=3 if use_wgmma else 5,
        num_sms=min(78, torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count),
        num_ctas_per_sm=1,
        use_tma=use_wgmma,
        use_warp_spec=use_wgmma,
        smem_reuse_mode="last_stage" if use_wgmma else "all_stages",
        use_stream_k=True,
        use_fp32_stream_k_reduce=True,
    )


@pytest.mark.parametrize("mma_type", ["mma", "wgmma"])
@pytest.mark.parametrize("bias", [0.0, 0.0625, 8.0])
@pytest.mark.parametrize(
    "shape_m,shape_n,nonzero,scaled",
    [(42, 6144, False, False), (49, 6144, True, True), (42, 512, True, False)],
)
@torch.inference_mode()
def test_dense_stream_k_fp32_reduction(mma_type, bias, shape_m, shape_n, nonzero, scaled):
    tuning = _dense_stream_k_tuning(mma_type)
    layer, weights = _make_dense_stream_k_layer(shape_n, nonzero=nonzero, bias=bias, scaled=scaled)
    inputs = torch.full((shape_m, 2048), 64.0, dtype=torch.float16, device="cuda")
    inputs[1::3] *= -1
    # CPU FP32 reference is independent of Humming and GPU TF32 settings.
    reference = inputs.cpu().float() @ weights.T + bias
    if not nonzero:
        assert torch.all(reference == bias)
    compute = dict(gemm_type="dense", use_f16_accum=False)
    for _ in range(3):
        output = layer(inputs, compute_config=compute, tuning_config=tuning)
        torch.testing.assert_close(output.cpu(), reference.half(), rtol=0, atol=0)
        assert not torch.count_nonzero(layer.locks).item()


@pytest.mark.parametrize("mma_type", ["mma", "wgmma"])
@torch.inference_mode()
def test_dense_stream_k_fp32_graph_and_streams(mma_type):
    tuning = _dense_stream_k_tuning(mma_type)
    compute = dict(gemm_type="dense", use_f16_accum=False)
    first, weights = _make_dense_stream_k_layer(512, nonzero=True, bias=0.0625)
    second, _ = _make_dense_stream_k_layer(512, nonzero=True, bias=0.0625)
    for shape_m in (1, 42, 49, 42):
        inputs = torch.full((shape_m, 2048), 64.0, dtype=torch.float16, device="cuda")
        output = first(inputs, compute_config=compute, tuning_config=tuning)
        reference = inputs.cpu().float() @ weights.T + 0.0625
        torch.testing.assert_close(output.cpu(), reference.half(), rtol=0, atol=0)

    for _ in range(3):
        first(inputs, compute_config=compute, tuning_config=tuning)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = first(inputs, compute_config=compute, tuning_config=tuning)
    for amplitude in (64, -64, 32):
        inputs.fill_(amplitude)
        graph.replay()
        reference = inputs.cpu().float() @ weights.T + 0.0625
        torch.testing.assert_close(output.cpu(), reference.half(), rtol=0, atol=0)

    first_stream, second_stream = torch.cuda.Stream(), torch.cuda.Stream()
    first_stream.wait_stream(torch.cuda.current_stream())
    second_stream.wait_stream(torch.cuda.current_stream())
    for _ in range(10):
        with torch.cuda.stream(first_stream):
            first_output = first(inputs, compute_config=compute, tuning_config=tuning)
        with torch.cuda.stream(second_stream):
            second_output = second(inputs, compute_config=compute, tuning_config=tuning)
    torch.cuda.synchronize()
    for result, layer in ((first_output, first), (second_output, second)):
        torch.testing.assert_close(result.cpu(), reference.half(), rtol=0, atol=0)
        assert not torch.count_nonzero(layer.locks).item()

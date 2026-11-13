import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from humming import dtypes, ops
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, TuningConfig, WeightScale2Type
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.forward import humming_forward
from humming.kernel.humming import HummingKernel
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)
from humming.testing.data import generate_random_tensor, generate_random_topk_ids
from humming.utils.smem import fits_device_smem

SHAPE_N = 1024
SHAPE_K = 1024
GROUPED_INPUT_SIZE = 128


SPECIAL_FEATURES = {
    "use_int_weight_scale",
    "use_fused_e8m0_scale",
    "use_packed_k_layout",
}


@pytest.mark.parametrize(
    "stages,iterations,shape_m,stream_k",
    [
        (2, 1, 1, False),  # first stage only
        (2, 3, 17, True),  # two-stage wrap
        (3, 2, 1, False),  # before the ring is full
        (3, 3, 17, True),  # exactly fills the ring
        (3, 4, 17, False),  # first steady-state refill
        (4, 9, 1, True),  # multiple ring wraps
    ],
)
def test_full_stage_prefetch_boundaries(stages, iterations, shape_m, stream_k):
    _check_full_stage_prefetch(
        shape_m=shape_m,
        shape_n=256,
        shape_k=128 * iterations,
        stages=stages,
        block_k=128,
        warp_k=64,
        stream_k=stream_k,
    )


def test_full_stage_prefetch_two_stage_producer():
    skip_if_unsupported(a_dtype="int8", mma_type="mma")
    _check_full_stage_prefetch(
        shape_m=193,
        shape_n=256,
        shape_k=64 * 3,
        stages=2,
        block_k=64,
        warp_k=64,
        stream_k=False,
        use_test_distribution=True,
        tuning_updates=dict(
            num_ctas_per_sm=1,
            num_sms=2,
        ),
    )


@pytest.mark.parametrize("stages,use_tma,use_warp_spec", [(33, False, True), (65, True, True)])
def test_full_stage_prefetch_deep_barriers(stages, use_tma, use_warp_spec):
    skip_if_unsupported(a_dtype="int8", mma_type="mma", use_tma=use_tma, use_warp_spec=use_warp_spec)
    _check_full_stage_prefetch(
        shape_m=193,
        shape_n=256,
        shape_k=64 * (2 * stages + 1),
        stages=stages,
        block_k=64,
        warp_k=64,
        stream_k=False,
        block_n=64,
        layer_updates={"b_dtype": "uint4"},
        use_test_distribution=True,
        tuning_updates=dict(
            use_tma=use_tma,
            use_warp_spec=use_warp_spec,
            num_ctas_per_sm=1,
            num_sms=2,
            producer_stage_unroll=1,
            consumer_stage_unroll=1,
            warp_shape=(16, 16 if use_warp_spec else 32, 64),
        ),
    )


def _check_full_stage_prefetch(
    shape_m,
    shape_n,
    shape_k,
    stages,
    block_k,
    warp_k,
    stream_k,
    *,
    block_n=128,
    layer_updates=None,
    gemm_type=GemmType.DENSE,
    use_test_distribution=False,
    tuning_updates=None,
):
    if not torch.cuda.is_available():
        pytest.skip("full-stage prefetch requires a CUDA device")
    layer_values = dict(
        shape_n=shape_n,
        shape_k=shape_k,
        a_dtype=dtypes.int8,
        b_dtype=dtypes.int8,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        use_packed_k_layout=False,
    )
    test_case = KernelTestCase(
        name="full-stage-prefetch",
        seed=2026,
        rtol=0.01 if use_test_distribution else 0.02,
        atol=0.05 if use_test_distribution else 0.1,
        layer_config=LayerConfig(**(layer_values | (layer_updates or {}))),
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=1 if gemm_type == GemmType.DENSE else 2,
    )
    runner = KernelTestRunner(test_case)
    torch.manual_seed(2026 + shape_m)
    if use_test_distribution:
        original = generate_random_tensor(
            (shape_m, shape_k),
            dtype=runner.layer_config.param_dtype,
            group_size=runner.layer_config.input_scale_group_size,
        )
    else:
        original = torch.randn((shape_m, shape_k), device="cuda", dtype=torch.bfloat16)
    topk_ids = None
    if gemm_type != GemmType.DENSE:
        topk_ids = generate_random_topk_ids(shape_m, runner.layer_config.num_experts, test_case.top_k)
    block_m = (tuning_updates or {}).get("block_shape", (16, block_n, block_k))[0]
    problem_inputs, route_tensors, output_ids = runner._prepare_problem(shape_m, original, topk_ids, block_m)
    inputs_ref, inputs, input_scale, input_scale_2 = runner.prepare_inputs(problem_inputs)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = runner.make_reference(inputs_ref, topk_ids, output_ids)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    tuning = dict(
        mma_type="mma",
        block_shape=(16, block_n, block_k),
        warp_shape=(16, 32, warp_k),
        num_stages=stages,
        num_ctas_per_sm=1 if block_k >= 512 else 2,
        use_stream_k=stream_k,
        use_warp_spec=False,
        use_tma=False,
    )
    tuning.update(tuning_updates or {})
    resource_tuning = TuningConfig(**{key: value for key, value in tuning.items() if key != "num_sms"})
    if not fits_device_smem(runner.layer_config, runner.compute_config, resource_tuning):
        pytest.skip("fixed full-stage-prefetch test configuration exceeds this device's SMEM limit")
    has_native_atomic_add = current_device.sm_version >= 90 or runner.layer_config.c_dtype != dtypes.bfloat16
    check_bitwise = not (stream_k and has_native_atomic_add)
    configs_pair = HummingKernel.prepare_kernels(
        runner.layer_config.to_str(),
        runner.compute_config.to_str(),
        [(0, 1 << 30, tuning | {"use_full_stage_prefetch": value}) for value in (False, True)],
    ).reshape(-1, 4)
    outputs, smem_sizes = [], []
    for configs in configs_pair:
        kernel = HummingKernel._id2kernel[configs[2].item()]
        kernel.assert_smem_size_matches_estimate()
        smem_sizes.append(ops.get_kernel_smem_size(kernel.kernel_id))
        output = torch.zeros(
            (test_case.effective_shape_m(shape_m), shape_n),
            device=inputs.device,
            dtype=dtypes.torch_dtype_map[runner.layer_config.c_dtype],
        )
        if gemm_type == GemmType.GROUPED_CONTIGUOUS:
            output = output[: inputs.shape[0]]
        locks = torch.zeros((2048,), dtype=torch.int32, device=inputs.device) if stream_k else None

        def run(configs=configs, output=output, locks=locks):
            return ops.launch_kernel(
                configs=configs,
                inputs=inputs,
                **runner.kernel_tensors,
                **route_tensors,
                input_scale=input_scale,
                input_scale_2=input_scale_2,
                outputs=output,
                locks=locks,
                top_k=test_case.top_k,
            )

        for _ in range(10):
            run()
            runner._assert_close(output[output_ids], reference, use_stream_k=stream_k)
            if outputs and check_bitwise:
                torch.testing.assert_close(output[output_ids], outputs[0][output_ids], rtol=0, atol=0)
        outputs.append(output)
    torch.cuda.synchronize()
    assert smem_sizes[0] == smem_sizes[1]


def _layer_config(**kwargs) -> LayerConfig:
    return LayerConfig(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        c_dtype=dtypes.bfloat16,
        **kwargs,
    )


def _kernel_case(
    required_features: tuple[str, ...],
    name: str,
    layer_config: LayerConfig,
    gemm_type: GemmType = GemmType.DENSE,
) -> tuple[tuple[str, ...], KernelTestCase]:
    is_dense = gemm_type == GemmType.DENSE
    test_case = KernelTestCase(
        name=name,
        layer_config=layer_config,
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=1 if is_dense else 2,
        seed=2026,
    )
    return required_features, test_case


SPECIAL_WEIGHT_CASES = (
    _kernel_case(
        required_features=(),
        name="nvfp4-a16-dense",
        layer_config=_layer_config(
            a_dtype=dtypes.bfloat16,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_2_type=WeightScale2Type.TENSOR,
        ),
    ),
    _kernel_case(
        required_features=(),
        name="nvfp4-a16-indexed",
        layer_config=_layer_config(
            a_dtype=dtypes.bfloat16,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_2_type=WeightScale2Type.TENSOR,
            num_experts=8,
        ),
        gemm_type=GemmType.INDEXED,
    ),
    _kernel_case(
        required_features=("use_int_weight_scale",),
        name="int-weight-scale-int8",
        layer_config=_layer_config(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=64,
            weight_scale_group_size_n=1,
        ),
    ),
    _kernel_case(
        required_features=("use_int_weight_scale",),
        name="int-weight-scale-int4",
        layer_config=_layer_config(
            a_dtype=dtypes.int4,
            b_dtype=dtypes.uint3,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=64,
            weight_scale_group_size_n=1,
        ),
    ),
    _kernel_case(
        required_features=("use_int_weight_scale",),
        name="odd-bit-packed-k-fallback",
        layer_config=_layer_config(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint5,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            weight_scale_group_size_n=1,
            sm_version=90,
        ),
    ),
    _kernel_case(
        required_features=("use_fused_e8m0_scale",),
        name="fused-e8m0-tensor-secondary-grouped-input",
        layer_config=_layer_config(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=GROUPED_INPUT_SIZE,
            weight_scale_group_size=64,
        ),
    ),
    _kernel_case(
        required_features=("use_fused_e8m0_scale",),
        name="fused-e8m0-channel-secondary",
        layer_config=_layer_config(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            weight_scale_group_size=64,
            weight_scale_2_type=WeightScale2Type.CHANNEL,
        ),
    ),
    _kernel_case(
        required_features=("use_packed_k_layout",),
        name="packed-k-fp8-grouped-input",
        layer_config=_layer_config(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            input_scale_group_size=GROUPED_INPUT_SIZE,
            weight_scale_group_size=128,
            sm_version=90,
        ),
    ),
    _kernel_case(
        required_features=("use_int_weight_scale", "use_packed_k_layout"),
        name="packed-k-int8-with-int-scale",
        layer_config=_layer_config(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint2,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            weight_scale_group_size_n=1,
            sm_version=90,
        ),
    ),
    _kernel_case(
        required_features=("use_packed_k_layout",),
        name="packed-k-zero-point",
        layer_config=_layer_config(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            has_zero_point=True,
            sm_version=90,
        ),
    ),
    *(
        _kernel_case(
            required_features=("use_packed_k_layout", "use_fused_e8m0_scale"),
            name=f"packed-k-fused-e8m0-{a_dtype}-gs{group_size}-as{input_group_size}-{gemm_type.value}",
            layer_config=_layer_config(
                a_dtype=a_dtype,
                b_dtype=dtypes.float4e2m1,
                bs_dtype=dtypes.float8e8m0,
                input_scale_group_size=input_group_size,
                weight_scale_group_size=group_size,
                weight_scale_2_type=WeightScale2Type.CHANNEL
                if input_group_size == 0
                else WeightScale2Type.TENSOR,
                num_experts=0 if gemm_type == GemmType.DENSE else 8,
                sm_version=90,
                use_packed_k_layout=True,
            ),
            gemm_type=gemm_type,
        )
        for a_dtype in (dtypes.float8e4m3, dtypes.int8)
        for group_size in (32, 64, 128)
        for input_group_size in (0, 128)
        for gemm_type in (GemmType.DENSE, GemmType.GROUPED_CONTIGUOUS)
    ),
)


@pytest.mark.filterwarnings(r"ignore:`torch\.jit\.script_method` is deprecated\. ")
@pytest.mark.parametrize("a_dtype", [dtypes.bfloat16, dtypes.int8], ids=str)
def test_forward_fullgraph(a_dtype):
    """Catch graph breaks in the forward path, including input quantization, and
    reuse the graph across token counts."""
    torch._dynamo.reset()
    skip_if_unsupported(a_dtype=a_dtype)
    config = _layer_config(
        a_dtype=a_dtype,
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
    )
    runner = KernelTestRunner(
        KernelTestCase(
            name="fullgraph",
            layer_config=config,
            compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        )
    )
    compute_config = runner.compute_config.to_str()
    locks = torch.zeros(1024, device="cuda", dtype=torch.int32)

    def forward(inputs):
        return humming_forward(
            config,
            inputs,
            **runner.kernel_tensors,
            locks=locks,
            compute_config=compute_config,
            tuning_config={},
        )

    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(forward, backend=counter, fullgraph=True, dynamic=True)
    for shape_m in (17, 257):
        torch.manual_seed(runner.test_case.seed + shape_m)
        # Match the numerical range used by KernelTestRunner: Stream-K stores
        # partial sums in the output dtype, so absolute error scales with inputs.
        inputs = generate_random_tensor((shape_m, SHAPE_K), dtype=torch.bfloat16, device="cuda")
        expected = (inputs.float() @ runner.weight_ref.T).to(torch.bfloat16)
        torch.testing.assert_close(forward(inputs), expected, rtol=0.01, atol=0.05)
        torch.testing.assert_close(compiled(inputs), expected, rtol=0.01, atol=0.05)
    assert counter.frame_count == 1


@pytest.mark.parametrize(
    "required_features,test_case",
    SPECIAL_WEIGHT_CASES,
    ids=[case.name for _, case in SPECIAL_WEIGHT_CASES],
)
def test_special_weight_path(required_features, test_case):
    config = test_case.layer_config
    if "use_fused_e8m0_scale" in required_features and get_default_mma_type(config) == MmaType.MXMMA:
        pytest.skip("fused E8M0 scale is not supported by MXMMA")

    for feature in required_features:
        assert getattr(config, feature) is True
    if "use_int_weight_scale" in required_features or "use_fused_e8m0_scale" in required_features:
        assert config.weight_scale_2_type != WeightScale2Type.NONE

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_special_weight_path_coverage():
    assert {feature for features, _ in SPECIAL_WEIGHT_CASES for feature in features} == SPECIAL_FEATURES

    configs_by_feature = {feature: [] for feature in SPECIAL_FEATURES}
    for features, case in SPECIAL_WEIGHT_CASES:
        for feature in features:
            configs_by_feature[feature].append(case.layer_config)

    int_scale_configs = configs_by_feature["use_int_weight_scale"]
    fused_configs = configs_by_feature["use_fused_e8m0_scale"]
    packed_k_configs = configs_by_feature["use_packed_k_layout"]
    assert {config.a_dtype for config in int_scale_configs} == {dtypes.int4, dtypes.int8}
    assert {config.weight_scale_2_type.name for config in fused_configs} == {"CHANNEL", "TENSOR"}
    assert {config.a_dtype for config in packed_k_configs} == {dtypes.float8e4m3, dtypes.int8}
    assert any(config.use_int_weight_scale for config in packed_k_configs)

    odd_bit_fallback = next(case.layer_config for _, case in SPECIAL_WEIGHT_CASES if "odd-bit" in case.name)
    assert odd_bit_fallback.b_dtype.num_bits % 2 == 1
    assert odd_bit_fallback.use_packed_k_layout is False


@pytest.mark.parametrize(
    "mma_type,block_m,chunk_rows,shape_n,gemm_type,use_tma,cta_group_size",
    (
        ("mma", 128, 96, 512, GemmType.DENSE, True, 1),
        ("mma", 128, 96, 504, GemmType.DENSE, True, 1),
        ("mma", 64, 32, 512, GemmType.INDEXED, False, 1),
        ("mma", 128, 96, 512, GemmType.GROUPED_CONTIGUOUS, True, 1),
        ("mma", 128, 96, 504, GemmType.GROUPED_MASKED, True, 1),
        ("mma", 64, 128, 512, GemmType.DENSE, True, 1),
        ("wgmma", 128, 0, 512, GemmType.DENSE, True, 1),
        ("wgmma", 128, 0, 512, GemmType.GROUPED_CONTIGUOUS, True, 1),
        ("wgmma", 128, 0, 504, GemmType.GROUPED_MASKED, True, 1),
        ("wgmma", 128, 96, 512, GemmType.DENSE, True, 1),
        ("wgmma", 128, 96, 504, GemmType.GROUPED_CONTIGUOUS, True, 1),
        ("umma", 48, 32, 512, GemmType.DENSE, True, 2),
        ("umma", 80, 64, 504, GemmType.DENSE, True, 1),
        ("umma", 80, 64, 512, GemmType.GROUPED_CONTIGUOUS, True, 1),
        ("umma", 48, 32, 504, GemmType.GROUPED_MASKED, True, 2),
        ("umma", 80, 64, 512, GemmType.INDEXED, False, 1),
        ("umma", 24, 64, 512, GemmType.DENSE, True, 1),
        ("umma", 48, 0, 512, GemmType.GROUPED_CONTIGUOUS, True, 2),
        ("umma", 128, 96, 512, GemmType.DENSE, True, 2),
    ),
)
def test_output_chunk_rows(
    mma_type, block_m, chunk_rows, shape_n, gemm_type, use_tma, cta_group_size, monkeypatch
):
    """Cover N slab stores, descriptor bounds, buffer reuse, scatter, and K reduction."""
    skip_if_unsupported(a_dtype=dtypes.bfloat16, mma_type=mma_type, use_tma=use_tma)
    layer = LayerConfig(
        shape_n=512,
        pad_shape_n=512 - shape_n,
        shape_k=1024,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        has_bias=True,
        num_experts=0 if gemm_type == GemmType.DENSE else 4,
    )
    if mma_type == "umma" and not layer.is_umma_supported:
        pytest.skip("UMMA requires SM10x or SM11x")
    is_umma = mma_type == "umma"
    is_mma = mma_type == "mma"
    # Keep N=256 slab coverage without exhausting registers or SM120 shared memory.
    block_k = 128 if is_umma else 64
    config = dict(
        mma_type=mma_type,
        block_shape=(block_m, 256, block_k),
        warp_shape=(block_m if is_umma else 64, 32 if is_umma else 64, 128 if is_umma else 64),
        num_stages=2 if is_mma else 3,
        num_sms=6,
        num_ctas_per_sm=1,
        use_tma=use_tma,
        use_stream_k=True,
        smem_reuse_mode="last_stage" if is_mma else "none",
        umma_cta_group_size=cta_group_size,
        output_chunk_rows=chunk_rows,
    )
    if mma_type == "wgmma":
        config["use_warp_spec"] = True
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: config)
    case = KernelTestCase(
        name="output-chunk-rows",
        layer_config=layer,
        compute_config=ComputeConfig(gemm_type=gemm_type),
        seed=2026,
    )
    shape_ms = (17, 13 * block_m + 1)
    results = KernelTestRunner(case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)

import pytest
import torch

from humming import dtypes, ops
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType
from humming.config.mma import get_default_mma_type
from humming.device import current_device
from humming.layer import HummingLayer
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)
from humming.utils.math import round_up

MIN_SHAPE_N = 64
MIN_SHAPE_K = 32
BLOCK_SCALED_SM_VERSION = (
    current_device.sm_version if current_device.sm_version // 10 in (10, 11, 12) else 120
)


def _case(
    name: str,
    *,
    shape_n: int,
    shape_k: int,
    pad_shape_n: int = 0,
    pad_shape_k: int = 0,
    a_dtype=dtypes.bfloat16,
    b_dtype=dtypes.uint4,
    bs_dtype=dtypes.bfloat16,
    as_dtype=None,
    input_scale_group_size: int = 0,
    input_quant_mode: str | None = None,
    weight_scale_type: str | None = None,
    weight_scale_group_size: int = 0,
    weight_scale_group_size_n: int = 0,
    mma_type: MmaType | None = None,
    sm_version: int | None = None,
) -> KernelTestCase:
    return KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=shape_n,
            shape_k=shape_k,
            pad_shape_n=pad_shape_n,
            pad_shape_k=pad_shape_k,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=dtypes.bfloat16,
            bs_dtype=bs_dtype,
            as_dtype=as_dtype,
            input_scale_group_size=input_scale_group_size,
            input_quant_mode=input_quant_mode,
            weight_scale_type=weight_scale_type,
            weight_scale_group_size=weight_scale_group_size,
            weight_scale_group_size_n=weight_scale_group_size_n,
            sm_version=90 if mma_type == MmaType.WGMMA else sm_version,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )


PROBLEM_SHAPE_CASES = (
    _case(
        "short-k-group-scale",
        shape_n=256,
        shape_k=64,
        weight_scale_group_size=64,
    ),
    _case(
        "minimum",
        shape_n=MIN_SHAPE_N,
        shape_k=MIN_SHAPE_K,
    ),
    _case(
        "non-umma-aligned-n",
        shape_n=192,
        shape_k=128,
    ),
    _case(
        "small-tile-boundary",
        shape_n=128,
        shape_k=64,
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint7,
    ),
    _case(
        "large-n-small-k",
        shape_n=8192,
        shape_k=MIN_SHAPE_K,
    ),
    _case(
        "small-n-large-k",
        shape_n=MIN_SHAPE_N,
        shape_k=8192,
        a_dtype=dtypes.int4,
        b_dtype=dtypes.uint3,
    ),
    _case(
        "large-rectangular",
        shape_n=4096,
        shape_k=8192,
    ),
    _case(
        "w6a8-group128",
        shape_n=6144,
        shape_k=4096,
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint6,
        weight_scale_group_size=128,
        weight_scale_group_size_n=1,
        mma_type=MmaType.WGMMA,
    ),
)


PAD_SHAPE_CASES = (
    _case(
        "minimum-pad-n",
        shape_n=128,
        shape_k=128,
        pad_shape_n=8,
    ),
    _case(
        "minimum-packable-pad-k",
        shape_n=128,
        shape_k=128,
        pad_shape_k=32,
    ),
    _case(
        "one-third-pad-n",
        shape_n=1024,
        shape_k=1024,
        pad_shape_n=344,
    ),
    _case(
        "one-third-pad-k",
        shape_n=1024,
        shape_k=1024,
        pad_shape_k=352,
    ),
    _case(
        "one-third-pad-nk",
        shape_n=1024,
        shape_k=1024,
        pad_shape_n=344,
        pad_shape_k=352,
        a_dtype=dtypes.int4,
        b_dtype=dtypes.uint3,
    ),
    _case(
        "maximum-pad-n",
        shape_n=1024,
        shape_k=1024,
        pad_shape_n=1024 - MIN_SHAPE_N,
    ),
    _case(
        "maximum-pad-k",
        shape_n=1024,
        shape_k=1024,
        pad_shape_k=1024 - MIN_SHAPE_K,
    ),
    _case(
        "maximum-pad-nk",
        shape_n=1024,
        shape_k=1024,
        pad_shape_n=1024 - MIN_SHAPE_N,
        pad_shape_k=1024 - MIN_SHAPE_K,
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint7,
    ),
)


# Cross dtype/scale coverage with K64 tile counts; aligned K alone misses MX scale tails.
NATIVE_SHAPE_CASES = tuple(
    _case(
        f"mxfp8-mxfp4-n{shape_n}-k{shape_k}",
        sm_version=BLOCK_SCALED_SM_VERSION,
        shape_n=shape_n,
        shape_k=shape_k,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        as_dtype=dtypes.float8e8m0,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=32,
        weight_scale_group_size=32,
        input_quant_mode="dynamic_group",
    )
    for shape_n, shape_k in ((128, 64), (384, 192), (256, 2816), (256, 2880), (256, 2944), (512, 4160))
) + (
    _case(
        "fp8-large-rectangular",
        shape_n=4096,
        shape_k=8192,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float8e4m3,
        input_quant_mode="static_tensor",
        weight_scale_type="tensor",
    ),
    _case(
        "fp8-uint4-n192-k2880",
        shape_n=192,
        shape_k=2880,
        a_dtype=dtypes.float8e4m3,
    ),
    _case(
        "fp8-uint2-n320-k192",
        shape_n=320,
        shape_k=192,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.uint2,
    ),
    _case(
        "fp8-channel-scale-n384-k192",
        shape_n=384,
        shape_k=192,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float8e4m3,
        bs_dtype=dtypes.float8e4m3,
    ),
    _case(
        "fp8-fp4-channel-scale-n4096-k2880",
        shape_n=4096,
        shape_k=2880,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
    ),
)


# FP8 GS128 activations x MXFP4 weights whose unpadded K ends in a partial input
# scale group, e.g. a TP-split MoE intermediate (2304 / 4 = 576) padded to 640.
PARTIAL_INPUT_GROUP_CASES = tuple(
    _case(
        f"fp8-gs128-mxfp4-n{shape_n}-k{shape_k}-pad{pad_shape_k}",
        shape_n=shape_n,
        shape_k=shape_k,
        pad_shape_k=pad_shape_k,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.float4e2m1,
        as_dtype=dtypes.float32,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=128,
        weight_scale_group_size=32,
        input_quant_mode="dynamic_group",
    )
    for shape_n, shape_k, pad_shape_k in ((512, 640, 64), (256, 384, 96), (256, 2944, 64))
)


SHAPE_CASES = PROBLEM_SHAPE_CASES + PAD_SHAPE_CASES + NATIVE_SHAPE_CASES + PARTIAL_INPUT_GROUP_CASES


@pytest.mark.parametrize("test_case", SHAPE_CASES, ids=str)
def test_shape(test_case):
    config = test_case.layer_config
    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_layer_preserves_explicit_padding_with_compact_weights():
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=72,
        shape_k=96,
        weight_config={"quant_method": "humming", "dtype": "uint4"},
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_from_unquantized(torch.randn(72, 96, device="cuda", dtype=torch.bfloat16))
    reference_weight = layer.weight_schema.dequant_tensors(dict(layer.state_dict())).float()
    layer.transform()
    assert layer.humming_config.shape_n == 256
    assert layer.humming_config.shape_k == 128
    assert layer.humming_config.pad_shape_n == 184
    assert layer.humming_config.pad_shape_k == 32
    assert layer.humming_config.weight_shape_n == 128
    assert layer.humming_config.weight_shape_k == 96
    for shape_m in (1, 17, 129):
        inputs = torch.randn(shape_m, 96, device="cuda", dtype=torch.bfloat16)
        expected = inputs.float() @ reference_weight.T
        outputs = layer(inputs)
        assert outputs.shape == (shape_m, 72)
        torch.testing.assert_close(outputs.float(), expected, rtol=0.02, atol=0.05)


@pytest.mark.parametrize("shape_n,shape_k", [(72, 96), (96, 96), (192, 192), (320, 1088)])
@pytest.mark.parametrize("use_tma", [False, True])
@pytest.mark.parametrize("use_stream_k", [False, True])
@pytest.mark.parametrize(
    "weight_mode",
    [
        "channel",
        "tensor",
        "group-zp",
        "group-zp-a8",
        "secondary",
        "raw",
        "input-group",
        "block",
        "mx",
        "mx-wide",
        "fused",
    ],
)
def test_partial_nk_tiles(shape_n, shape_k, use_tma, use_stream_k, weight_mode, monkeypatch):
    layer_values = dict(a_dtype=dtypes.bfloat16, b_dtype=dtypes.uint4, bs_dtype=dtypes.bfloat16)
    if weight_mode in ("group-zp", "group-zp-a8") and shape_n % 32:
        pytest.skip("packed source zero points require N aligned to 32")
    if weight_mode == "tensor":
        layer_values.update(weight_scale_type="tensor", bs_dtype=dtypes.float32)
    elif weight_mode == "group-zp":
        layer_values.update(weight_scale_group_size=32, has_zero_point=True)
    elif weight_mode == "group-zp-a8":
        layer_values.update(
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.uint3,
            weight_scale_group_size=32,
            has_zero_point=True,
        )
    elif weight_mode == "secondary":
        layer_values.update(weight_scale_group_size=32, weight_scale_2_type="channel", has_bias=True)
    elif weight_mode == "raw":
        layer_values.update(a_dtype=dtypes.float8e4m3, b_dtype=dtypes.float8e4m3)
    elif weight_mode == "input-group":
        layer_values.update(a_dtype=dtypes.float8e4m3, b_dtype=dtypes.float8e4m3, input_scale_group_size=128)
    elif weight_mode == "block":
        if shape_n % 64 or shape_k % 64:
            pytest.skip("block quantization requires complete source groups")
        layer_values.update(weight_scale_group_size=64, weight_scale_group_size_n=64, bs_dtype=dtypes.float32)
    elif weight_mode in ("mx", "mx-wide"):
        layer_values.update(
            a_dtype=dtypes.float4e2m1,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=32,
            weight_scale_group_size=32,
        )
    elif weight_mode == "fused":
        layer_values.update(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            weight_scale_group_size=32,
            use_fused_e8m0_scale=True,
        )
    group_size = layer_values.get("weight_scale_group_size", 0)
    if group_size and shape_k % group_size:
        pytest.skip("weight quantization requires complete source groups")
    padded_n = round_up(shape_n, 512 if weight_mode == "mx-wide" else 128)
    padded_k = round_up(shape_k, 128)
    layer = LayerConfig(
        shape_n=padded_n,
        shape_k=padded_k,
        pad_shape_n=padded_n - shape_n,
        pad_shape_k=padded_k - shape_k,
        c_dtype=dtypes.bfloat16,
        **layer_values,
    )
    mma_type = get_default_mma_type(layer)
    skip_if_unsupported(a_dtype=layer.a_dtype, mma_type=mma_type.value, use_tma=use_tma)
    if mma_type == MmaType.UMMA and not use_tma:
        pytest.skip("UMMA requires TMA weight loading")
    if mma_type == MmaType.UMMA:
        block_shape = (32, 128, 128)
        warp_shape = (32, 32, 128)
    else:
        block_shape = (16, 128, 128)
        warp_shape = (16, 32, 512 // layer.a_dtype.num_bits)
    if weight_mode == "mx-wide":
        block_shape = (block_shape[0], 512, block_shape[2])
    tuning = dict(
        block_shape=block_shape,
        warp_shape=warp_shape,
        num_stages=2 if weight_mode == "mx-wide" else 3,
        num_sms=6,
        num_ctas_per_sm=1,
        use_tma=use_tma,
        use_stream_k=use_stream_k,
        use_warp_spec=False,
    )
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: tuning)
    case = KernelTestCase(
        name=f"partial-nk-{weight_mode}",
        layer_config=layer,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )
    shape_ms = (17, 129)
    results = KernelTestRunner(case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)


@pytest.mark.parametrize("shape_n", [128, 384])
@pytest.mark.parametrize("use_stream_k", [False, True])
@pytest.mark.parametrize("use_tma_c", [False, True])
@pytest.mark.parametrize(
    "mma_type,use_tma_b",
    [("wgmma", False), ("wgmma", True), ("umma", True)],
)
def test_padding_only_n_cta(shape_n, use_stream_k, use_tma_c, mma_type, use_tma_b, monkeypatch):
    """Cover whole padding CTAs; run under memcheck to also detect invalid B reads."""
    skip_if_unsupported(a_dtype=dtypes.bfloat16, mma_type=mma_type, use_tma=True)
    is_umma = mma_type == "umma"
    layer = LayerConfig(
        shape_n=round_up(shape_n, 256),
        pad_shape_n=128,
        shape_k=1024,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        has_bias=True,
    )
    if is_umma and not layer.is_umma_supported:
        pytest.skip("UMMA requires SM10x or SM11x")

    block_k = 128 if is_umma else 64
    tuning = dict(
        mma_type=mma_type,
        block_shape=(64, 128, block_k),
        warp_shape=(64, 32, block_k),
        num_stages=3,
        num_sms=6,
        num_ctas_per_sm=1,
        use_tma=True,
        use_tma_b=use_tma_b,
        use_tma_c=use_tma_c,
        use_warp_spec=True,
        use_stream_k=use_stream_k,
        multi_cast_size_a=1 if is_umma else 2,
        umma_cta_group_size=2 if is_umma else 1,
    )
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: tuning)

    launch_kernel = ops.launch_kernel

    def launch_with_output_guard(**kwargs):
        outputs = kwargs["outputs"]
        # The padding CTA can write one full tile past the final output row.
        storage = outputs.new_full((outputs.numel() + 128,), 42)
        guarded_outputs = storage[: outputs.numel()].view_as(outputs)
        guarded_outputs.copy_(outputs)
        kwargs["outputs"] = guarded_outputs
        result = launch_kernel(**kwargs)
        guard = storage[outputs.numel() :]
        torch.testing.assert_close(guard, torch.full_like(guard, 42), rtol=0, atol=0)
        return result

    monkeypatch.setattr(ops, "launch_kernel", launch_with_output_guard)
    case = KernelTestCase(
        name="padding-only-n-cta",
        layer_config=layer,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )
    shape_ms = (17, 129)
    results = KernelTestRunner(case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)
    assert all(result.tuning_config.use_tma_b == use_tma_b for result in results)
    assert all(result.tuning_config.use_tma_c == use_tma_c for result in results)


FP8_TOKEN_MXFP4 = dict(
    a_dtype=dtypes.float8e4m3,
    b_dtype=dtypes.float4e2m1,
    as_dtype=dtypes.float32,
    bs_dtype=dtypes.float8e8m0,
    input_quant_mode="dynamic_token",
    weight_scale_group_size=32,
)
FP8_TOKEN_FP8_CHANNEL = dict(
    a_dtype=dtypes.float8e4m3,
    b_dtype=dtypes.float8e4m3,
    as_dtype=dtypes.float32,
    bs_dtype=dtypes.bfloat16,
    input_quant_mode="dynamic_token",
    weight_scale_group_size=0,
)
FUSED_MXFP4_LAYOUT = dict(use_fused_e8m0_scale=True, use_packed_k_layout=False, input_scale_group_size=0)
FP8_CHANNEL_LAYOUT = dict(
    use_fused_e8m0_scale=False, use_packed_k_layout=False, input_scale_group_size=0, weight_scale_group_size=0
)


def _padded_k_case(
    name,
    unpadded_k,
    shape_k,
    block_k,
    warp_k=64,
    num_stages=3,
    layer_values=None,
    expected_layout=FUSED_MXFP4_LAYOUT,
    **tuning_values,
):
    padding = dict(shape_k=shape_k, pad_shape_k=shape_k - unpadded_k)
    layer_values = FP8_TOKEN_MXFP4 | (layer_values or {}) | padding
    tile = dict(block_shape=(64, 128, block_k), warp_shape=(64, 32, warp_k), num_stages=num_stages)
    return pytest.param(layer_values, tile | tuning_values, expected_layout, id=name)


PADDED_K_CASES = (
    _padded_k_case("whole-padded-block", 288, 384, block_k=64),
    _padded_k_case("k-split-warpgroups", 288, 384, block_k=128, num_stages=4),
    _padded_k_case("split-issue-wait", 288, 384, block_k=64, wgmma_split_issue_wait=True),
    _padded_k_case("single-unpadded-block", 32, 128, block_k=64),
    _padded_k_case("aligned-unpadded-k", 256, 384, block_k=64),
    _padded_k_case("last-block-stage-0", 224, 256, block_k=64),
    _padded_k_case("last-block-last-stage", 160, 192, block_k=64),
    _padded_k_case(
        "ss-k-batch",
        288,
        512,
        block_k=256,
        warp_k=128,
        layer_values=FP8_TOKEN_FP8_CHANNEL,
        expected_layout=FP8_CHANNEL_LAYOUT,
    ),
    _padded_k_case(
        "ss-k-batch-split-issue-wait",
        288,
        512,
        block_k=256,
        warp_k=128,
        layer_values=FP8_TOKEN_FP8_CHANNEL,
        expected_layout=FP8_CHANNEL_LAYOUT,
        wgmma_split_issue_wait=True,
    ),
    _padded_k_case(
        "packed-k",
        288,
        512,
        block_k=256,
        warp_k=128,
        layer_values=dict(use_packed_k_layout=True),
        expected_layout=FUSED_MXFP4_LAYOUT | dict(use_packed_k_layout=True),
    ),
    _padded_k_case(
        "promotion-excluded",
        288,
        384,
        block_k=64,
        layer_values=dict(
            input_quant_mode="dynamic_group", input_scale_group_size=128, use_packed_k_layout=False
        ),
        expected_layout=FUSED_MXFP4_LAYOUT | dict(input_scale_group_size=128),
    ),
    _padded_k_case("stream-k-excluded", 288, 384, block_k=64, num_stages=5, use_stream_k=True),
    _padded_k_case("warp-spec-excluded", 288, 384, block_k=128, num_stages=4, use_warp_spec=True),
)


@pytest.mark.parametrize("layer_values,tuning_values,expected_layout", PADDED_K_CASES)
def test_padded_k_slabs(layer_values, tuning_values, expected_layout, monkeypatch):
    skip_if_unsupported(a_dtype=dtypes.float8e4m3, mma_type="wgmma", use_tma=True)
    layer = LayerConfig(shape_n=512, c_dtype=dtypes.bfloat16, **layer_values)
    for name, value in expected_layout.items():
        assert getattr(layer, name) == value, (name, getattr(layer, name))
    config = (
        dict(
            mma_type="wgmma",
            num_sms=6,
            num_ctas_per_sm=1,
            use_tma=True,
            use_stream_k=False,
            use_warp_spec=False,
        )
        | tuning_values
    )
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: config)
    case = KernelTestCase(
        name="padded-k-slabs",
        layer_config=layer,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )
    # On 6 SMs, M=17 makes stream-K split K mid-tile; M=129 runs full data-parallel waves.
    shape_ms = (17, 129)
    results = KernelTestRunner(case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)


@pytest.mark.parametrize("mma_type,k_warps", [("mma", 1), ("mma", 2), ("umma", 1)])
@pytest.mark.parametrize("use_stream_k", [False, True])
@pytest.mark.parametrize("unpadded_k", [32, 96, 288, 352, 256])
@pytest.mark.parametrize("weight_mode", ["group-zp", "fp8-channel", "fp8-input-group", "int8-fused", "mxfp4"])
def test_mma_umma_padded_k_slabs(mma_type, k_warps, use_stream_k, unpadded_k, weight_mode, monkeypatch):
    is_umma = mma_type == "umma"
    if is_umma and current_device.sm_version // 10 not in (10, 11):
        pytest.skip("UMMA requires SM10x or SM11x")
    if is_umma and weight_mode in ("fp8-input-group", "int8-fused"):
        pytest.skip("Software scale promotion and fused INT8 are MMA cases")
    if not is_umma and weight_mode == "mxfp4":
        pytest.skip("Block-scaled FP4 uses UMMA or MXMMA")
    layer_values = dict(a_dtype=dtypes.bfloat16, b_dtype=dtypes.uint4, bs_dtype=dtypes.bfloat16)
    if weight_mode == "group-zp":
        layer_values.update(weight_scale_group_size=32, has_zero_point=True)
    elif weight_mode == "int8-fused":
        layer_values.update(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            weight_scale_group_size=32,
            use_fused_e8m0_scale=True,
        )
    elif weight_mode == "mxfp4":
        layer_values.update(
            a_dtype=dtypes.float4e2m1,
            b_dtype=dtypes.float4e2m1,
            bs_dtype=dtypes.float8e8m0,
            input_scale_group_size=32,
            weight_scale_group_size=32,
        )
    else:
        layer_values.update(a_dtype=dtypes.float8e4m3, b_dtype=dtypes.float8e4m3)
        if weight_mode == "fp8-input-group":
            layer_values.update(input_scale_group_size=128)
    skip_if_unsupported(a_dtype=layer_values["a_dtype"], mma_type=mma_type)
    block_k = 256 if weight_mode == "mxfp4" else 128
    padded_k = 256 if unpadded_k == 256 else round_up(unpadded_k, block_k) + block_k
    layer = LayerConfig(
        shape_n=256,
        shape_k=padded_k,
        pad_shape_k=padded_k - unpadded_k,
        c_dtype=dtypes.bfloat16,
        has_bias=True,
        **layer_values,
    )
    if not is_umma and layer.use_block_scaled_mma:
        pytest.skip("This device selects MXMMA for the FP8 scale layout")
    warp_k = block_k // k_warps
    tuning = dict(
        mma_type=mma_type,
        block_shape=(32, 128, block_k),
        # Single-K-warp specialization and the split-K-warp fallback must both remain correct.
        warp_shape=(32 if is_umma else 16, 32, warp_k),
        num_stages=3,
        num_sms=6,
        num_ctas_per_sm=1,
        use_tma=is_umma,
        use_warp_spec=is_umma,
        use_stream_k=use_stream_k,
        umma_cta_group_size=2 if is_umma and unpadded_k in (32, 352) else 1,
        umma_num_dequant_warpgroups=2 if is_umma and unpadded_k == 96 else 1,
    )
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: tuning)
    case = KernelTestCase(
        name="mma-umma-padded-k-slabs",
        layer_config=layer,
        compute_config=ComputeConfig(gemm_type=GemmType.DENSE),
        seed=2026,
    )
    shape_ms = (17, 129)
    results = KernelTestRunner(case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)


@pytest.mark.skipif(not current_device.is_ppu, reason="Requires PPU AIU loads")
@pytest.mark.parametrize("shape_n,shape_k", [(72, 96), (192, 192), (128, 128), (72, 128), (128, 96)])
@pytest.mark.parametrize("use_raw_weight", [False, True])
@pytest.mark.parametrize("use_stream_k", [False, True])
@pytest.mark.parametrize("gemm_type", list(GemmType))
def test_ppu_padding(shape_n, shape_k, use_raw_weight, use_stream_k, gemm_type, monkeypatch):
    padded_n = round_up(shape_n, 128)
    padded_k = round_up(shape_k, 128)
    layer = LayerConfig(
        shape_n=padded_n,
        shape_k=padded_k,
        pad_shape_n=padded_n - shape_n,
        pad_shape_k=padded_k - shape_k,
        num_experts=0 if gemm_type == GemmType.DENSE else 4,
        a_dtype=dtypes.int8 if use_raw_weight else dtypes.bfloat16,
        b_dtype=dtypes.int8 if use_raw_weight else dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        has_bias=True,
    )
    assert layer.use_raw_weight == use_raw_weight
    tuning = dict(
        mma_type="mma",
        block_shape=(16, 128, 128),
        warp_shape=(16, 32, 64 if use_stream_k else 128),
        num_stages=3,
        num_sms=6,
        num_ctas_per_sm=1,
        use_tma=False,
        use_warp_spec=False,
        use_stream_k=use_stream_k,
    )
    monkeypatch.setenv("HUMMING_TEST_TUNING_SOURCE", "heuristic")
    monkeypatch.setattr("humming.testing.tuning.get_heuristics_config", lambda *args, **kwargs: tuning)
    case = KernelTestCase(
        name="ppu-padding",
        layer_config=layer,
        compute_config=ComputeConfig(gemm_type=gemm_type),
        top_k=1 if gemm_type == GemmType.DENSE else 2,
        seed=2026,
        atol=0.2,
    )
    shape_ms = (1, 17, 129)
    results = KernelTestRunner(case).run(shape_ms)
    assert_kernel_test_shape_coverage(results, shape_ms)


def test_shape_case_coverage():
    problem_configs = {case.name: case.layer_config for case in PROBLEM_SHAPE_CASES}
    minimum = problem_configs["minimum"]
    tile_boundary = problem_configs["small-tile-boundary"]
    large_n = problem_configs["large-n-small-k"]
    large_k = problem_configs["small-n-large-k"]
    large_rectangular = problem_configs["large-rectangular"]

    assert (minimum.shape_n, minimum.shape_k) == (MIN_SHAPE_N, MIN_SHAPE_K)
    assert large_n.shape_n >= 8192 and large_n.shape_k == MIN_SHAPE_K
    assert large_k.shape_n == MIN_SHAPE_N and large_k.shape_k >= 8192
    assert large_rectangular.shape_n >= 4096 and large_rectangular.shape_k >= 4096

    assert (minimum.a_dtype.num_bits, minimum.b_dtype.num_bits) == (16, 4)
    assert (tile_boundary.a_dtype.num_bits, tile_boundary.b_dtype.num_bits) == (8, 7)
    assert (large_k.a_dtype.num_bits, large_k.b_dtype.num_bits) == (4, 3)

    pad_configs = {case.name: case.layer_config for case in PAD_SHAPE_CASES}
    pad_modes = {(bool(config.pad_shape_n), bool(config.pad_shape_k)) for config in pad_configs.values()}
    assert pad_modes == {(True, False), (False, True), (True, True)}

    one_third = pad_configs["one-third-pad-nk"]
    assert one_third.pad_shape_n / one_third.shape_n == pytest.approx(1 / 3, abs=0.01)
    assert one_third.pad_shape_k / one_third.shape_k == pytest.approx(1 / 3, abs=0.02)
    assert (one_third.a_dtype.num_bits, one_third.b_dtype.num_bits) == (4, 3)

    maximum = pad_configs["maximum-pad-nk"]
    assert maximum.shape_n - maximum.pad_shape_n == MIN_SHAPE_N
    assert maximum.shape_k - maximum.pad_shape_k == MIN_SHAPE_K
    assert (maximum.a_dtype.num_bits, maximum.b_dtype.num_bits) == (8, 7)

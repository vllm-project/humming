"""MXFP4 W4A8 coverage with grouped FP8 inputs."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.config.mma import get_default_mma_type, get_dtype_compatibility_error
from humming.schema.compressed_tensors import CompressedTensorsInputSchema
from humming.schema.humming import HummingInputSchema, HummingWeightSchema
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

SHAPE_N = 1024
SHAPE_K = 1024
INPUT_GROUP_SIZE = 128
WEIGHT_GROUP_SIZE = 32
NUM_EXPERTS = 8


def _layer_config(
    *,
    shape_n: int = SHAPE_N,
    shape_k: int = SHAPE_K,
    num_experts: int = 0,
    use_fused_e8m0_scale: bool | None = None,
    a_dtype=dtypes.float8e4m3,
    as_dtype=None,
    input_scale_group_size: int = INPUT_GROUP_SIZE,
) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=a_dtype,
        as_dtype=as_dtype,
        b_dtype=dtypes.float4e2m1,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        input_scale_group_size=input_scale_group_size,
        weight_scale_group_size=WEIGHT_GROUP_SIZE,
        sm_version=90,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
    )


def _case(
    name: str,
    *,
    shape_n: int = SHAPE_N,
    shape_k: int = SHAPE_K,
    gemm_type: GemmType = GemmType.DENSE,
    num_experts: int = NUM_EXPERTS,
    top_k: int = 2,
    use_m_major_input_scale: bool = False,
    use_fused_e8m0_scale: bool | None = None,
    a_dtype=dtypes.float8e4m3,
    as_dtype=None,
    input_scale_group_size: int = INPUT_GROUP_SIZE,
) -> KernelTestCase:
    is_dense = gemm_type == GemmType.DENSE
    return KernelTestCase(
        name=name,
        layer_config=_layer_config(
            shape_n=shape_n,
            shape_k=shape_k,
            num_experts=0 if is_dense else num_experts,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
            a_dtype=a_dtype,
            as_dtype=as_dtype,
            input_scale_group_size=input_scale_group_size,
        ),
        compute_config=ComputeConfig(gemm_type=gemm_type, use_m_major_input_scale=use_m_major_input_scale),
        top_k=1 if is_dense else top_k,
        seed=2026,
        atol=0.5 if use_m_major_input_scale else 0.05,
    )


MXFP4_CASES = (
    (
        False,
        _case(
            "mxfp4-a16-dense",
            a_dtype=dtypes.bfloat16,
            input_scale_group_size=0,
        ),
    ),
    (
        False,
        _case(
            "mxfp4-a16-indexed",
            gemm_type=GemmType.INDEXED,
            a_dtype=dtypes.bfloat16,
            input_scale_group_size=0,
        ),
    ),
    (True, _case("mxfp4-grouped-fp8-dense-auto")),
    (
        True,
        _case(
            "mxfp4-grouped-fp8-g32-dense-n64-k64",
            shape_n=2880,
            shape_k=2880,
            as_dtype=dtypes.float32,
            input_scale_group_size=32,
        ),
    ),
    (False, _case("mxfp4-grouped-fp8-dense-nonfused", use_fused_e8m0_scale=False)),
    (True, _case("mxfp4-grouped-fp8-indexed-auto", gemm_type=GemmType.INDEXED)),
    (
        True,
        _case(
            "mxfp4-grouped-fp8-grouped-contiguous-auto",
            gemm_type=GemmType.GROUPED_CONTIGUOUS,
        ),
    ),
    (
        True,
        _case(
            "mxfp4-grouped-fp8-grouped-masked-auto",
            gemm_type=GemmType.GROUPED_MASKED,
        ),
    ),
    *(
        (
            True,
            _case(
                f"mxfp4-m-major-e{experts}-n{shape_n}-k{shape_k}",
                shape_n=shape_n,
                shape_k=shape_k,
                gemm_type=GemmType.GROUPED_CONTIGUOUS,
                num_experts=experts,
                top_k=8,
                use_m_major_input_scale=True,
            ),
        )
        for experts, shape_n, shape_k in (
            (8, 1024, 1024),
            (8, 1088, 1024),
            (33, 1024, 1024),
            (128, 1024, 1024),
            (32, 4096, 6144),
            (32, 6144, 2048),
        )
    ),
)


@pytest.mark.parametrize(
    "expected_fused,test_case",
    MXFP4_CASES,
    ids=[case.name for _, case in MXFP4_CASES],
)
def test_mxfp4(expected_fused, test_case):
    config = test_case.layer_config
    assert config.use_fused_e8m0_scale is expected_fused
    assert config.is_group_weight_scale
    assert config.is_tensor_weight_scale_2 is expected_fused
    if test_case.uses_m_major_input_scale:
        assert config.use_packed_k_layout

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=get_default_mma_type(config).value)
    results = KernelTestRunner(test_case).run()
    if test_case.uses_m_major_input_scale:
        assert all(
            result.tuning_values.get("use_packed_k_layout", config.use_packed_k_layout) for result in results
        )
    assert_kernel_test_shape_coverage(results)


def test_mxfp4_case_coverage():
    assert {expected_fused for expected_fused, _ in MXFP4_CASES} == {False, True}
    assert {case.compute_config.gemm_type for _, case in MXFP4_CASES} == {
        GemmType.DENSE,
        GemmType.INDEXED,
        GemmType.GROUPED_CONTIGUOUS,
        GemmType.GROUPED_MASKED,
    }
    assert {case.layer_config.a_dtype for _, case in MXFP4_CASES} == {
        dtypes.bfloat16,
        dtypes.float8e4m3,
    }


@pytest.mark.parametrize("checkpoint_format", ["mxfp4-pack-quantized", "float-quantized"])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_mxfp4_input_schema_compatibility(checkpoint_format, group_size, monkeypatch):
    monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=90))
    weight = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=WEIGHT_GROUP_SIZE,
    )
    inputs = CompressedTensorsInputSchema(
        format=checkpoint_format,
        type="float",
        num_bits=8,
        dynamic=True,
        group_size=group_size,
    ).to_humming_schema(torch.bfloat16)
    assert inputs.input_scale_dtype is None
    assert inputs.is_compatible_with(weight, torch.bfloat16) == (group_size in (32, INPUT_GROUP_SIZE))


@pytest.mark.parametrize("sm_version", [89, 90, 100, 103, 107, 110, 120, 121])
@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16", "float8e4m3", "float8e5m2", "float8e3m4"])
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
@pytest.mark.parametrize("bs_dtype", [None, "bfloat16", "float8e4m3", "float8e8m0"])
@pytest.mark.parametrize("use_group_input_scale", [False, True])
def test_fp4_weight_schema_compatibility(
    sm_version, a_dtype, group_size, bs_dtype, use_group_input_scale, monkeypatch
):
    monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=sm_version))
    activation_dtype = dtypes.DataType.from_str(a_dtype)
    has_fp8_activation = activation_dtype.num_bits == 8
    input_group_size = group_size if has_fp8_activation and use_group_input_scale else 0
    inputs = HummingInputSchema(a_dtype=activation_dtype, input_scale_group_size=input_group_size)
    weight = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1,
        bs_dtype=bs_dtype,
        weight_scale_group_size=group_size,
    )
    has_supported_groups = not has_fp8_activation or group_size >= 32
    has_supported_dtype = a_dtype != "float8e3m4" or sm_version in (100, 103, 120, 121)
    has_supported_scales = True
    if a_dtype == "float8e3m4" and sm_version in (100, 103):
        has_supported_scales = group_size == 32 and bs_dtype == "float8e8m0"
    resolved_bs_dtype = dtypes.DataType.from_str(bs_dtype) if bs_dtype is not None else dtypes.bfloat16
    dtype_error = get_dtype_compatibility_error(activation_dtype, dtypes.bfloat16, resolved_bs_dtype)
    is_supported = all([
        has_supported_groups,
        has_supported_dtype,
        has_supported_scales,
        dtype_error is None
    ])
    assert inputs.is_compatible_with(weight, torch.bfloat16) == is_supported


@pytest.mark.parametrize("sm_version", [89, 90, 100, 103, 107, 110, 120, 121])
@pytest.mark.parametrize("operand_dtype", ["float4e2m1", "float4e0m3"])
@pytest.mark.parametrize("bs_dtype", ["bfloat16", "float8e4m3", "float8e8m0"])
@pytest.mark.parametrize("as_dtype", ["float32", "float8e4m3", "float8e8m0"])
def test_nvfp4_input_schema_compatibility(sm_version, operand_dtype, bs_dtype, as_dtype, monkeypatch):
    monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=sm_version))
    inputs = HummingInputSchema(
        a_dtype=operand_dtype,
        input_scale_group_size=16,
        input_scale_dtype=as_dtype,
    )
    weight = HummingWeightSchema(
        b_dtype=operand_dtype,
        bs_dtype=bs_dtype,
        weight_scale_group_size=16,
    )
    has_supported_dtype = sm_version >= 100
    if operand_dtype == "float4e0m3":
        has_supported_dtype = sm_version in (100, 103, 120, 121)
    is_supported = has_supported_dtype and bs_dtype in ("float8e4m3", "float8e8m0") and as_dtype == bs_dtype
    assert inputs.is_compatible_with(weight, torch.bfloat16) == is_supported


@pytest.mark.parametrize("sm_version", [89, 90, 100, 103, 107, 110, 120, 121])
@pytest.mark.parametrize("a_dtype", ["float16", "bfloat16"])
def test_undocumented_weight_schema_compatibility(sm_version, a_dtype, monkeypatch):
    monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=sm_version))
    inputs = HummingInputSchema(a_dtype=a_dtype)
    weight = HummingWeightSchema(b_dtype=dtypes.float8e3m4, weight_scale_group_size=32)
    # Wider activations use software dequantization, so no undocumented hardware operand is required.
    output_dtype = dtypes.float16 if a_dtype == "float16" else dtypes.bfloat16
    assert inputs.is_compatible_with(weight, dtypes.torch_dtype_map[output_dtype])


E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
# (case, activation dtype, expected activation dtype, expected weight_scale_2 type)
E8M0_RANGE_CASES = [
    ("in_range", "float8e4m3", "float8e4m3", "TENSOR"),
    ("fused_shards", "float8e4m3", "float8e4m3", "CHANNEL"),
    ("n_padded", "float8e4m3", "float8e4m3", "TENSOR"),
    ("zero_block", "float8e4m3", "float8e4m3", "TENSOR"),
    ("row_spread", "float8e4m3", "bfloat16", "NONE"),
    ("fused_shards", "int8", "int8", "CHANNEL"),
    ("in_range", "int8", "bfloat16", "NONE"),
]


def _e8m0_range_weights(case: str):
    """E2M1 codes and E8M0 scale codes whose spread stresses the fused E8M0 range."""
    shape_n = 128 if case == "n_padded" else 512
    shape_k = 1024
    generator = torch.Generator().manual_seed(0)
    codes = torch.randint(1, 16, (shape_n, shape_k), generator=generator)
    codes[codes == 8] = 1
    scales = torch.full((shape_n, shape_k // WEIGHT_GROUP_SIZE), 120, dtype=torch.uint8)
    if case == "in_range":
        scales[0, 1] = 120 - 11
    elif case == "fused_shards":
        scales[shape_n // 2 :] = 120 - 17
    elif case == "n_padded":
        scales.fill_(127 - 16)
    elif case == "zero_block":
        scales[0, 1] = 0
        codes[0, WEIGHT_GROUP_SIZE : 2 * WEIGHT_GROUP_SIZE] = 0
    elif case == "row_spread":
        scales[3, 0] = 124
        scales[3, 1] = 108
    return codes, scales


@pytest.mark.parametrize(("case", "a_dtype", "expected_a_dtype", "expected_scale_2"), E8M0_RANGE_CASES)
def test_mxfp4_fused_e8m0_scale_range_is_exact(case, a_dtype, expected_a_dtype, expected_scale_2):
    """Fused E8M0 scaling only covers 11 (FP8) or 3 (INT8) binades below the
    largest scale; layers whose scales spread further must pick a config that
    still reproduces every checkpoint weight exactly."""
    skip_if_unsupported(a_dtype=dtypes.DataType.from_str(a_dtype), mma_type="wgmma")
    from humming.config import InputQuantizationMode, WeightScale2Type
    from humming.forward import humming_forward
    from humming.transform import prepare_and_transform_humming_tensors

    codes, scales = _e8m0_range_weights(case)
    shape_n, shape_k = codes.shape
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).contiguous()
    weight_schema = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=WEIGHT_GROUP_SIZE,
    )
    input_schema = HummingInputSchema(
        a_dtype=a_dtype,
        input_scale_dtype=dtypes.float32,
        input_quant_mode=InputQuantizationMode.DynamicToken,
    )
    tensors = {
        "weight": packed.view(torch.int32).cuda(),
        "weight_scale": scales.cuda().view(torch.float8_e8m0fnu),
    }
    with pytest.warns(UserWarning) if expected_a_dtype != a_dtype else contextlib.nullcontext():
        config, tensors = prepare_and_transform_humming_tensors(
            tensors=tensors,
            shape_n=shape_n,
            shape_k=shape_k,
            weight_schema=weight_schema,
            input_schema=input_schema,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
    assert str(config.a_dtype) == expected_a_dtype
    assert config.weight_scale_2_type == WeightScale2Type[expected_scale_2]

    identity = torch.eye(shape_k, dtype=torch.bfloat16, device="cuda")
    outputs = humming_forward(
        config,
        inputs=identity,
        weight=tensors["weight"],
        weight_scale=tensors.get("weight_scale"),
        weight_scale_2=tensors.get("weight_scale_2"),
        locks=torch.zeros(1024, dtype=torch.int32, device="cuda"),
        compute_config={"gemm_type": "dense"},
    )
    group_scales = torch.exp2(scales.float() - 127).repeat_interleave(WEIGHT_GROUP_SIZE, dim=1)
    expected = E2M1_VALUES[codes] * group_scales
    torch.testing.assert_close(outputs[:, :shape_n].float().T.cpu(), expected, rtol=0, atol=0)

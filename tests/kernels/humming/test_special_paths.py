import json

import pytest
import torch

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, WeightScale2Type
from humming.forward import humming_forward
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

SHAPE_N = 1024
SHAPE_K = 1024
GROUPED_INPUT_SIZE = 128

SPECIAL_FEATURES = {
    "use_int_weight_scale",
    "use_fused_e8m0_scale",
    "use_packed_k_layout",
}


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
            mma_type=MmaType.WGMMA,
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
            mma_type=MmaType.WGMMA,
        ),
    ),
    _kernel_case(
        required_features=("use_int_weight_scale", "use_packed_k_layout"),
        name="packed-k-int8-with-int-scale",
        layer_config=_layer_config(
            a_dtype=dtypes.int8,
            b_dtype=dtypes.uint4,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            weight_scale_group_size_n=1,
            mma_type=MmaType.WGMMA,
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
            mma_type=MmaType.WGMMA,
        ),
    ),
)


@pytest.mark.parametrize("weight_scale_type", ["tensor", "channel"])
@pytest.mark.parametrize("config_format", ["str", "dict"])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_fp8_weight_only_forward_fullgraph(weight_scale_type, config_format, backend):
    """Compile the public forward path used by vLLM's FP8 weight-only layers."""
    skip_if_unsupported(a_dtype=dtypes.bfloat16)
    config = _layer_config(
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.float8e4m3,
        bs_dtype=dtypes.bfloat16,
        weight_scale_type=weight_scale_type,
    )
    runner = KernelTestRunner(
        KernelTestCase(name="compiled-fp8-weight-only", layer_config=config, compute_config=ComputeConfig())
    )
    compute_config = {"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "dense"}
    if config_format == "str":
        compute_config = json.dumps(compute_config)
    inputs = torch.randn(17, SHAPE_K, device="cuda", dtype=torch.bfloat16)
    locks = torch.zeros(1024, device="cuda", dtype=torch.int32)

    def forward(inputs):
        return humming_forward(
            config, inputs, **runner.kernel_tensors, locks=locks, compute_config=compute_config
        )

    expected = (inputs.float() @ runner.weight_ref.T).to(torch.bfloat16)
    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    torch.testing.assert_close(compiled(inputs), expected, rtol=0.01, atol=0.05)


@pytest.mark.parametrize(
    "required_features,test_case",
    SPECIAL_WEIGHT_CASES,
    ids=[case.name for _, case in SPECIAL_WEIGHT_CASES],
)
def test_special_weight_path(required_features, test_case):
    config = test_case.layer_config
    if "use_fused_e8m0_scale" in required_features and config.mma_type == MmaType.MXMMA:
        pytest.skip("fused E8M0 scale is not supported by MXMMA")

    for feature in required_features:
        assert getattr(config, feature) is True
    if "use_int_weight_scale" in required_features or "use_fused_e8m0_scale" in required_features:
        assert config.weight_scale_2_type != WeightScale2Type.NONE

    skip_if_unsupported(a_dtype=config.a_dtype, mma_type=config.mma_type.value)
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

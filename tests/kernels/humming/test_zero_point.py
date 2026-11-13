import dataclasses

import pytest
import torch

from humming import dtypes, ops
from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.config.mma import get_default_mma_type, get_dtype_compatibility_error, get_mma_b_dtype
from humming.device import current_device
from humming.testing import (
    KernelTestCase,
    KernelTestRunner,
    assert_kernel_test_shape_coverage,
    skip_if_unsupported,
)

SHAPE_N = 2048
SHAPE_K = 2048
WEIGHT_SCALE_GROUP_SIZE = 64

A_DTYPES = (
    "float16",
    "bfloat16",
    "float8e4m3",
    "float8e5m2",
    "float8e3m4",
    "int8",
    "int4",
)
B_DTYPES = tuple(f"uint{num_bits}" for num_bits in range(1, 9))
C_DTYPES = ("float16", "bfloat16")


def _output_dtypes(a_dtype):
    output_dtypes = tuple(dtypes.DataType.from_str(dtype) for dtype in C_DTYPES)
    return tuple(
        c_dtype
        for c_dtype in output_dtypes
        if get_dtype_compatibility_error(a_dtype, c_dtype, c_dtype) is None
    )


def _supports_integer_zero_point(a_dtype, b_dtype) -> bool:
    if b_dtype.num_bits > a_dtype.num_bits:
        return False
    if a_dtype.is_integer_type:
        return b_dtype.num_bits < a_dtype.num_bits
    mma_b_dtype = get_mma_b_dtype(a_dtype, b_dtype, True, current_device.sm_version)
    return b_dtype.num_bits <= mma_b_dtype.mantissa_bits + 1


def _make_cases() -> list[KernelTestCase]:
    cases = []
    compute_config = ComputeConfig(gemm_type=GemmType.DENSE)

    for is_fp_zero_point in (False, True):
        for a_dtype_str in A_DTYPES:
            a_dtype = dtypes.DataType.from_str(a_dtype_str)
            if is_fp_zero_point and a_dtype.num_bits != 16:
                continue

            for b_dtype_str in B_DTYPES:
                b_dtype = dtypes.DataType.from_str(b_dtype_str)
                if not _supports_integer_zero_point(a_dtype, b_dtype):
                    continue

                for c_dtype in _output_dtypes(a_dtype):
                    zero_point_kind = "fp-zp" if is_fp_zero_point else "int-zp"
                    layer_config = LayerConfig(
                        shape_n=SHAPE_N,
                        shape_k=SHAPE_K,
                        a_dtype=a_dtype,
                        b_dtype=b_dtype,
                        c_dtype=c_dtype,
                        bs_dtype=c_dtype,
                        input_scale_group_size=0,
                        weight_scale_group_size=64,
                        use_int_weight_scale=False,
                        has_zero_point=True,
                        is_fp_zero_point=is_fp_zero_point,
                    )
                    case = KernelTestCase(
                        name=(f"{zero_point_kind}-{a_dtype}-{layer_config.b_dtype}-{c_dtype}"),
                        layer_config=layer_config,
                        compute_config=compute_config,
                        seed=2026,
                    )
                    cases.append(case)

    return cases


ZERO_POINT_CASES = _make_cases()
ZERO_POINT_CASES += [
    dataclasses.replace(
        case,
        name=f"{case.name}-channel",
        layer_config=dataclasses.replace(
            case.layer_config, weight_scale_group_size=0, weight_scale_type="channel"
        ),
    )
    for case in ZERO_POINT_CASES
    if case.layer_config.a_dtype == dtypes.int8
    and case.layer_config.b_dtype in (dtypes.uint4, dtypes.uint7)
    and case.layer_config.c_dtype == dtypes.bfloat16
]


@pytest.mark.parametrize("test_case", ZERO_POINT_CASES, ids=str)
def test_zero_point(test_case):
    skip_if_unsupported(
        a_dtype=test_case.layer_config.a_dtype,
        b_dtype=get_mma_b_dtype(
            test_case.layer_config.a_dtype,
            test_case.layer_config.b_dtype,
            test_case.layer_config.has_zero_point,
            current_device.sm_version,
        ),
        mma_type=get_default_mma_type(test_case.layer_config).value,
    )
    results = KernelTestRunner(test_case).run()
    assert_kernel_test_shape_coverage(results)


def test_zero_point_case_coverage():
    integer_cases = [case for case in ZERO_POINT_CASES if not case.layer_config.is_fp_zero_point]
    floating_cases = [case for case in ZERO_POINT_CASES if case.layer_config.is_fp_zero_point]

    assert {str(case.layer_config.a_dtype) for case in integer_cases} == set(A_DTYPES)
    assert {case.layer_config.b_dtype.num_bits for case in integer_cases} == set(range(1, 9))
    assert {str(case.layer_config.c_dtype) for case in ZERO_POINT_CASES} == set(C_DTYPES)
    assert {str(case.layer_config.a_dtype) for case in floating_cases} == {"float16", "bfloat16"}
    assert {case.layer_config.b_dtype.num_bits for case in floating_cases} == set(range(1, 9))
    assert all(case.layer_config.has_zero_point for case in ZERO_POINT_CASES)
    assert {case.layer_config.weight_scale_group_size for case in ZERO_POINT_CASES} == {0, 64}


def test_repack_accepts_strided_zero_point():
    """A zero point packed along N arrives as a transposed view from some loaders."""
    shape_n, shape_k, group_size = 256, 256, 64
    weight = torch.randint(0, 256, (shape_n, shape_k), dtype=torch.uint8, device="cuda")
    weight = weight.view(torch.int32)
    zero_point = torch.randint(0, 256, (shape_k // group_size, shape_n), dtype=torch.uint8, device="cuda")
    zero_point = zero_point.view(torch.int32).t()
    assert not zero_point.is_contiguous()

    def repack(zp):
        return ops.repack_weight(
            inputs=weight,
            zero_point=zp,
            weight_bits=8,
            activation_bits=16,
            is_weight_packed=True,
            should_preprocess_for_int2fp=True,
            should_preprocess_with_zp=True,
            group_size_zp=group_size,
        )

    assert torch.equal(repack(zero_point), repack(zero_point.contiguous()))

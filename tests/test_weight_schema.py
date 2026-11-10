import dataclasses

import pytest
import torch

from humming import dtypes, ops
from humming.schema import humming as schema_module
from humming.schema.humming import HummingWeightSchema

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _reference_conversion(schema, tensors):
    """Original whole-tensor conversion, retained as a bit-exact and memory reference."""
    num_bits = schema.b_dtype.num_bits
    has_integer_zero_point = schema.has_zero_point and not schema.is_fp_zero_point
    if not schema.b_dtype.is_integer_type or not 2 <= num_bits <= 8 or has_integer_zero_point:
        return schema, tensors

    weight = ops.unpack_weight(tensors["weight"], num_bits)
    if torch.any(weight == 0):
        return schema, tensors

    midpoint = 1 << (num_bits - 1)
    output_tensors = tensors.copy()
    if schema.has_zero_point:
        zero_point = tensors["zero_point"]
        centered_zero_point = zero_point - midpoint
        if not torch.equal(centered_zero_point.float() + midpoint, zero_point.float()):
            return schema, tensors
        output_tensors["zero_point"] = centered_zero_point

    weight = weight - midpoint
    weight = weight.abs() | ((weight < 0).to(torch.int32) << (num_bits - 1))
    output_tensors["weight"] = ops.pack_weight(weight, num_bits)
    weight_dtype = dtypes.FloatingPointType(num_bits=num_bits, exponent_bits=0, mantissa_bits=num_bits - 1)
    return dataclasses.replace(schema, b_dtype=weight_dtype), output_tensors


def _make_tensors(num_bits, shape):
    # Exercise every nonzero code, including both signs and the midpoint.
    count = 1
    for size in shape:
        count *= size
    weight = torch.arange(count, dtype=torch.int32, device="cuda") % ((1 << num_bits) - 1) + 1
    return {
        "weight": ops.pack_weight(weight.view(shape), num_bits),
        "weight_scale": torch.ones((), device="cuda"),
    }


@pytest.mark.parametrize("num_bits", range(2, 9))
@pytest.mark.parametrize("shape", [(5, 96), (3, 5, 96)])
@pytest.mark.parametrize("zero_point_dtype", [None, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("chunk_size", [64, 1 << 20])
def test_symmetric_conversion_matches_reference(monkeypatch, num_bits, shape, zero_point_dtype, chunk_size):
    monkeypatch.setattr(schema_module, "_SYMMETRIC_CONVERSION_CHUNK_SIZE", chunk_size)
    schema = HummingWeightSchema(
        b_dtype=f"int{num_bits}", has_zero_point=zero_point_dtype is not None, is_fp_zero_point=True
    )
    tensors = _make_tensors(num_bits, shape)
    if zero_point_dtype is not None:
        tensors["zero_point"] = torch.tensor([0, 1 << (num_bits - 1)], dtype=zero_point_dtype, device="cuda")
    original = {key: value.clone() for key, value in tensors.items()}

    expected_schema, expected = _reference_conversion(schema, tensors)
    actual_schema, actual = schema.convert_symmetric_integer_weight(tensors)

    assert actual_schema == expected_schema
    assert actual_schema.b_dtype.is_floating_point_type
    assert actual is not tensors
    assert actual["weight"].shape == tensors["weight"].shape
    assert actual["weight"].dtype == torch.int32
    assert actual["weight_scale"] is tensors["weight_scale"]
    for key in tensors:
        assert torch.equal(actual[key], expected[key])
        assert torch.equal(tensors[key], original[key])


@pytest.mark.parametrize("num_bits", range(2, 9))
@pytest.mark.parametrize("zero_index", [0, 63, 64, 479, 480, -1])
def test_symmetric_conversion_zero_keeps_all_experts(monkeypatch, num_bits, zero_index):
    monkeypatch.setattr(schema_module, "_SYMMETRIC_CONVERSION_CHUNK_SIZE", 64)
    schema = HummingWeightSchema(b_dtype=f"uint{num_bits}")
    weight = torch.ones((3, 5, 96), dtype=torch.int32, device="cuda")
    weight.view(-1)[zero_index] = 0
    tensors = {"weight": ops.pack_weight(weight, num_bits)}
    original = tensors["weight"].clone()

    actual_schema, actual = schema.convert_symmetric_integer_weight(tensors)

    assert actual_schema is schema
    assert actual is tensors
    assert torch.equal(tensors["weight"], original)


@pytest.mark.parametrize("num_bits", range(2, 9))
@pytest.mark.parametrize("zero_point_dtype", [torch.float16, torch.bfloat16])
def test_symmetric_conversion_rejects_lossy_zero_point(monkeypatch, num_bits, zero_point_dtype):
    monkeypatch.setattr(schema_module, "_SYMMETRIC_CONVERSION_CHUNK_SIZE", 64)
    schema = HummingWeightSchema(b_dtype=f"uint{num_bits}", has_zero_point=True, is_fp_zero_point=True)
    tensors = _make_tensors(num_bits, (3, 5, 96))
    tensors["zero_point"] = torch.tensor([0.1], dtype=zero_point_dtype, device="cuda")
    original = {key: value.clone() for key, value in tensors.items()}

    expected_schema, expected = _reference_conversion(schema, tensors)
    actual_schema, actual = schema.convert_symmetric_integer_weight(tensors)

    assert expected_schema is actual_schema is schema
    assert expected is actual is tensors
    for key in tensors:
        assert torch.equal(tensors[key], original[key])


@pytest.mark.parametrize(
    "schema",
    [
        HummingWeightSchema(b_dtype="uint1"),
        HummingWeightSchema(b_dtype="uint16"),
        HummingWeightSchema(b_dtype="float16"),
        HummingWeightSchema(b_dtype="uint4", has_zero_point=True),
    ],
)
def test_symmetric_conversion_ineligible_schema_does_not_unpack(monkeypatch, schema):
    def unexpected_unpack(*args, **kwargs):
        pytest.fail("ineligible schema must not unpack weights")

    monkeypatch.setattr(ops, "unpack_weight", unexpected_unpack)
    tensors = {}
    actual_schema, actual = schema.convert_symmetric_integer_weight(tensors)
    assert actual_schema is schema
    assert actual is tensors


def _measure_peak(conversion, schema, tensors):
    torch.cuda.synchronize()
    initial_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = conversion(schema, tensors)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - initial_memory
    return result, peak


@pytest.mark.parametrize("has_zero", [False, True])
def test_symmetric_conversion_peak_memory(has_zero, record_property):
    schema = HummingWeightSchema(b_dtype="uint3")
    weight = torch.ones((8, 512, 2048), dtype=torch.int32, device="cuda")
    if has_zero:
        weight[-1, -1, -1] = 0
    tensors = {"weight": ops.pack_weight(weight, 3)}
    del weight

    # Warm the pack/unpack kernels and allocator before either measurement.
    for conversion in (_reference_conversion, HummingWeightSchema.convert_symmetric_integer_weight):
        conversion(schema, tensors)
    (expected_schema, expected), reference_peak = _measure_peak(_reference_conversion, schema, tensors)
    (actual_schema, actual), chunked_peak = _measure_peak(
        HummingWeightSchema.convert_symmetric_integer_weight, schema, tensors
    )
    record_property("reference_peak_bytes", reference_peak)
    record_property("chunked_peak_bytes", chunked_peak)

    assert actual_schema == expected_schema
    assert torch.equal(actual["weight"], expected["weight"])
    assert chunked_peak < reference_peak // 4, (chunked_peak, reference_peak)
    if has_zero:
        assert actual_schema is schema
        assert actual is tensors

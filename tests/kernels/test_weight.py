import math

import pytest
import torch

from humming import ops


def _get_dtype_str(kind, num_bits):
    return f"float{num_bits}e0m{num_bits - 1}" if kind == "e0m" else f"{kind}{num_bits}"


def _convert_reference(codes, source_kind, target_kind, num_bits):
    midpoint = 1 << (num_bits - 1)
    values = []
    for code in codes:
        code &= 2 * midpoint - 1
        if source_kind == "e0m":
            value = -(code - midpoint) if code >= midpoint else code
        elif source_kind == "int":
            value = code - 2 * midpoint if code >= midpoint else code
        else:
            value = code - midpoint

        if target_kind == "e0m":
            magnitude = min(abs(value), midpoint - 1)
            value = magnitude + (midpoint if value < 0 and magnitude else 0)
        elif target_kind == "uint":
            value += midpoint
        # uint32 codes are stored in an int32 tensor.
        values.append(value - (1 << 32) if value >= 1 << 31 else value)
    return values


def _pack_reference(codes, num_bits):
    mask = (1 << num_bits) - 1
    words = []
    buffer = 0
    buffered_bits = 0
    for code in codes:
        buffer |= (code & mask) << buffered_bits
        buffered_bits += num_bits
        while buffered_bits >= 32:
            word = buffer & 0xFFFFFFFF
            words.append(word - (1 << 32) if word >= 1 << 31 else word)
            buffer >>= 32
            buffered_bits -= 32
    assert buffered_bits == 0
    return torch.tensor(words, dtype=torch.int32, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_bits", [*range(1, 9), 16, 31, 32])
@pytest.mark.parametrize("source_kind", ["int", "uint", "e0m"])
@pytest.mark.parametrize("target_kind", ["int", "uint", "e0m"])
def test_convert_weight(num_bits, source_kind, target_kind):
    midpoint = 1 << (num_bits - 1)
    if num_bits <= 8:
        codes = list(range(2 * midpoint))
    else:
        codes = [0, 1, midpoint - 1, midpoint, midpoint + 1, 2 * midpoint - 1]
    if source_kind == "int":
        codes += [-midpoint, -1]
    # Repeat to exercise multiple blocks and a partial final block.
    num_codes = 32 * 257
    codes = (codes * math.ceil(num_codes / len(codes)))[:num_codes]
    inputs = _pack_reference(codes, num_bits).reshape(257, num_bits)
    expected = _convert_reference(codes, source_kind, target_kind, num_bits)
    expected = _pack_reference(expected, num_bits).reshape_as(inputs)
    source_dtype = _get_dtype_str(source_kind, num_bits)
    target_dtype = _get_dtype_str(target_kind, num_bits)

    actual = ops.convert_weight(inputs, source_dtype, target_dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    outputs = torch.empty_like(inputs)
    assert ops.convert_weight(inputs, source_dtype, target_dtype, outputs=outputs) is outputs
    torch.testing.assert_close(outputs, expected, rtol=0, atol=0)
    original = inputs.clone()
    can_convert = ops.convert_weight(inputs, source_dtype, target_dtype, check_only=True)
    invalid_code = midpoint if source_kind == "int" else 0
    can_lose_precision = source_kind != "e0m" and target_kind == "e0m"
    has_invalid_code = any(code & (2 * midpoint - 1) == invalid_code for code in codes)
    expected_check = int(not (can_lose_precision and has_invalid_code))
    assert can_convert.shape == ()
    assert can_convert.dtype == torch.int32
    assert can_convert.device == inputs.device
    assert can_convert.item() == expected_check
    check_output = torch.empty((), dtype=torch.int32, device=inputs.device)
    actual_check = ops.convert_weight(
        inputs, source_dtype, target_dtype, check_only=True, outputs=check_output
    )
    assert actual_check is check_output
    assert actual_check.item() == expected_check
    torch.testing.assert_close(inputs, original, rtol=0, atol=0)

    safe_codes = [code for code in codes if code & (2 * midpoint - 1) != invalid_code]
    safe_codes = (safe_codes * 32)[: len(safe_codes) // 32 * 32 or 32]
    safe_inputs = _pack_reference(safe_codes, num_bits)
    assert ops.convert_weight(safe_inputs, source_dtype, target_dtype, check_only=True).item() == 1

    input_ptr = inputs.data_ptr()
    assert ops.convert_weight(inputs, source_dtype, target_dtype, inplace=True) is inputs
    assert inputs.data_ptr() == input_ptr
    torch.testing.assert_close(inputs, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_bits", range(2, 9))
def test_convert_packed_weight_roundtrip(num_bits):
    inputs = torch.arange(256, dtype=torch.int32, device="cuda").reshape(2, 128)
    inputs &= (1 << num_bits) - 1
    packed = ops.pack_weight(inputs, num_bits)
    signed = ops.convert_weight(packed, f"uint{num_bits}", f"int{num_bits}")
    restored = ops.convert_weight(signed, f"int{num_bits}", f"uint{num_bits}")
    torch.testing.assert_close(restored, packed, rtol=0, atol=0)
    torch.testing.assert_close(ops.unpack_weight(restored, num_bits), inputs, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_bits", [1, 3, 4, 5, 7, 16, 31, 32])
@pytest.mark.parametrize("num_groups", [0, 1, 31, 32, 33, 255, 256, 257, 511, 512, 513])
def test_convert_weight_check_boundaries(num_bits, num_groups):
    midpoint = 1 << (num_bits - 1)
    num_codes = num_groups * 32 // math.gcd(num_bits, 32)
    codes = [midpoint] * num_codes
    inputs = _pack_reference(codes, num_bits)
    outputs = torch.empty((), dtype=torch.int32, device="cuda")
    source_dtype = f"uint{num_bits}"
    target_dtype = f"float{num_bits}e0m{num_bits - 1}"
    check = ops.convert_weight(inputs, source_dtype, target_dtype, check_only=True, outputs=outputs)
    assert check.item() == 1
    if num_codes:
        codes[-1] = 0
        inputs.copy_(_pack_reference(codes, num_bits))
        check = ops.convert_weight(inputs, source_dtype, target_dtype, check_only=True, outputs=outputs)
        assert check.item() == 0
        codes[-1] = midpoint
        inputs.copy_(_pack_reference(codes, num_bits))
        check = ops.convert_weight(inputs, source_dtype, target_dtype, check_only=True, outputs=outputs)
        assert check.item() == 1
    converted = ops.convert_weight(inputs, source_dtype, target_dtype, inplace=True)
    assert converted is inputs
    assert torch.all(converted == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_bits", range(1, 9))
@pytest.mark.parametrize("source_dtype", [torch.int8, torch.uint8, torch.int32])
@pytest.mark.parametrize("target_dtype", [torch.int8, torch.uint8, torch.int32])
@pytest.mark.parametrize("shape", [(2, 32), (2, 3, 544), (0, 32)])
def test_pack_unpack_weight_storage(num_bits, source_dtype, target_dtype, shape):
    values = torch.arange(math.prod(shape), dtype=torch.int32, device="cuda") - 128
    inputs = values.to(source_dtype).reshape(shape)
    packed_shape = (*shape[:-1], shape[-1] * num_bits // 32)
    expected_packed = _pack_reference(inputs.flatten().tolist(), num_bits).reshape(packed_shape)
    packed = ops.pack_weight(inputs, num_bits)
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)

    expected = (inputs.to(torch.int32) & ((1 << num_bits) - 1)).to(target_dtype)
    actual = ops.unpack_weight(packed, num_bits, dtype=target_dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if inputs.numel():
        pack_output = torch.empty_like(packed)
        assert ops.pack_weight(inputs, num_bits, outputs=pack_output) is pack_output
        torch.testing.assert_close(pack_output, expected_packed, rtol=0, atol=0)
        unpack_output = torch.empty_like(expected)
        assert ops.unpack_weight(packed, num_bits, outputs=unpack_output) is unpack_output
        torch.testing.assert_close(unpack_output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_bits", [16, 31, 32])
def test_pack_unpack_weight_int32(num_bits):
    codes = [0, 1, -1, -(1 << 31), (1 << 31) - 1, 1 << (num_bits - 1), (1 << num_bits) - 1, 0] * 4
    inputs = torch.tensor(codes, dtype=torch.int64, device="cuda").to(torch.int32)
    expected_packed = _pack_reference(codes, num_bits)
    packed = ops.pack_weight(inputs, num_bits)
    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    expected = (inputs.to(torch.int64) & ((1 << num_bits) - 1)).to(torch.int32)
    torch.testing.assert_close(ops.unpack_weight(packed, num_bits), expected, rtol=0, atol=0)

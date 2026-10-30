"""Non-MMA weight kernels on ROCm: pack/unpack, dequant, quant, process_mxfp4, repack."""

import itertools

import pytest
import torch

from humming import dtypes, ops
from humming.utils.weight import dequantize_weight, quantize_weight

E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
UINT32_MASK = 0xFFFFFFFF


def _reference_pack(values: torch.Tensor, bits: int) -> torch.Tensor:
    values_per_word_group = values.view(-1, 32).to(torch.int64).cpu()
    packed = torch.zeros(values_per_word_group.shape[0], bits, dtype=torch.int64)
    for value_index in range(32):
        bit_index = value_index * bits
        word_index = bit_index // 32
        bit_offset = bit_index % 32
        packed[:, word_index] |= (values_per_word_group[:, value_index] << bit_offset) & UINT32_MASK
        if bit_offset + bits > 32:
            packed[:, word_index + 1] |= values_per_word_group[:, value_index] >> (32 - bit_offset)
    return packed.view(-1)


@pytest.mark.parametrize("bits", range(1, 9))
def test_pack_unpack_exact(bits):
    torch.manual_seed(bits)
    values = torch.randint(0, 2**bits, (256, 1024), dtype=torch.int32, device="cuda")
    packed = ops.pack_weight(values, bits)
    unpacked = ops.unpack_weight(packed, bits)
    packed_words = packed.view(-1).cpu().to(torch.int64) & UINT32_MASK
    assert torch.equal(packed_words, _reference_pack(values, bits))
    assert torch.equal(unpacked.view(-1), values.view(-1))


@pytest.mark.parametrize(
    ("exponent_bits", "mantissa_bits", "torch_dtype"),
    [(4, 3, torch.float8_e4m3fn), (5, 2, torch.float8_e5m2)],
)
def test_dequant_fp8_matches_torch(exponent_bits, mantissa_bits, torch_dtype):
    codes = torch.arange(256, dtype=torch.int32, device="cuda")
    dequantized = ops.dequant_weight(codes.clone(), exponent_bits, mantissa_bits, True)
    reference = codes.to(torch.uint8).view(torch_dtype).float()
    is_finite = reference.isfinite()
    assert torch.equal(dequantized[is_finite], reference[is_finite])


def test_dequant_e2m1_table():
    magnitudes = torch.tensor(E2M1_MAGNITUDES, device="cuda")
    expected = torch.cat([magnitudes, -magnitudes])
    codes = torch.arange(16, dtype=torch.int32, device="cuda")
    assert torch.equal(ops.dequant_weight(codes, 2, 1, True), expected)


QUANT_CASES = [
    ("uint4", "bfloat16", 128, False, 0.15),
    ("uint4", "float16", 128, True, 0.15),
    ("uint8", "bfloat16", 64, False, 0.01),
    ("float8e4m3", "bfloat16", 128, False, 0.05),
    ("float8e5m2", "float16", 0, False, 0.1),
    ("float4e2m1", "float8e8m0", 32, False, 0.2),
    ("float4e2m1", "float8e4m3", 16, False, 0.2),
]


QUANT_CASE_FIELDS = ("dtype_name", "scale_dtype_name", "group_size", "has_zero_point", "max_rel_err")


@pytest.mark.parametrize(QUANT_CASE_FIELDS, QUANT_CASES)
@pytest.mark.parametrize("source_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_quant_round_trip(
    source_dtype, dtype_name, scale_dtype_name, group_size, has_zero_point, max_rel_err
):
    torch.manual_seed(0)
    weight = torch.randn(256, 512, dtype=source_dtype, device="cuda")
    dtype = dtypes.DataType.from_str(dtype_name)
    scale_dtype = dtypes.DataType.from_str(scale_dtype_name)
    quantized, scales, zero_points, global_scale = quantize_weight(
        weight, dtype, scale_dtype, group_size, has_zero_point=has_zero_point
    )
    dequantized = dequantize_weight(quantized, scales, zero_points, global_scale, dtype)
    relative_error = ((dequantized - weight.float()).norm() / weight.float().norm()).item()
    assert relative_error < max_rel_err


def _reference_process_mxfp4(words: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Rescale each e2m1 nibble by 2**-delta and round to nearest (ties away from zero).

    Input -0 (code 8) is canonicalized to +0; values that round to zero keep their sign (code 8).
    """
    grid = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float64)
    word_groups = words.cpu().view(-1, 4).to(torch.int64) & UINT32_MASK
    shifts = deltas.cpu().view(-1).to(torch.int64)
    result = torch.zeros_like(word_groups)
    for nibble_index in range(8):
        nibbles = (word_groups >> (4 * nibble_index)) & 0xF
        nibbles = torch.where(nibbles == 8, 0, nibbles)
        signs = nibbles & 8
        magnitude_codes = nibbles & 7
        scaled = grid[magnitude_codes] * (2.0 ** (-shifts)).unsqueeze(1)
        distances = (scaled.unsqueeze(-1) - grid).abs()
        nearest_codes = 7 - distances.flip(-1).argmin(-1)
        new_codes = torch.where(shifts.unsqueeze(1) > 0, nearest_codes, magnitude_codes)
        result |= (signs | new_codes) << (4 * nibble_index)
    return result


def test_process_mxfp4_matches_reference():
    torch.manual_seed(0)
    num_groups = 4096
    words = torch.randint(-(2**31), 2**31 - 1, (num_groups * 4,), dtype=torch.int32, device="cuda")
    deltas = torch.randint(0, 12, (num_groups,), dtype=torch.uint8, device="cuda")
    processed = ops.process_mxfp4_w4a8_weight(words, deltas)
    processed_words = processed.cpu().view(-1, 4).to(torch.int64) & UINT32_MASK
    assert torch.equal(processed_words, _reference_process_mxfp4(words, deltas))


def _packed_value_histogram(packed: torch.Tensor, bits: int) -> torch.Tensor:
    words = packed.view(-1).to(torch.int64) & UINT32_MASK
    values = torch.stack([(words >> (bits * index)) & (2**bits - 1) for index in range(32 // bits)]).view(-1)
    return torch.bincount(values, minlength=2**bits)


REPACK_CASES = list(itertools.product([2, 4, 8], [16, 8], [0, 1, 2, 3]))


@pytest.mark.parametrize(("weight_bits", "activation_bits", "interleave_mode"), REPACK_CASES)
def test_repack(weight_bits, activation_bits, interleave_mode):
    torch.manual_seed(0)
    values = torch.randint(0, 2**weight_bits, (256, 1024), dtype=torch.int32, device="cuda")
    original_values = values.clone()
    repacked_runs = [
        ops.repack_weight(values, weight_bits, activation_bits, False, interleave_mode=interleave_mode)
        for _ in range(3)
    ]
    packed_values = ops.pack_weight(values, weight_bits)
    repacked_from_packed = ops.repack_weight(
        packed_values, weight_bits, activation_bits, True, interleave_mode=interleave_mode
    )
    assert torch.equal(values, original_values)
    assert all(torch.equal(repacked_runs[0], run) for run in repacked_runs[1:])
    assert torch.equal(repacked_runs[0], repacked_from_packed)

    # Modes 1/2 bit-interleave values inside a word (except w8a8), so only those keep values whole.
    keeps_values_whole = interleave_mode in (0, 3) or (weight_bits == 8 and activation_bits == 8)
    if keeps_values_whole:
        expected_histogram = torch.bincount(values.view(-1).long(), minlength=2**weight_bits)
        assert torch.equal(_packed_value_histogram(repacked_runs[0], weight_bits), expected_histogram)

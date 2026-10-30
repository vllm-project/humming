"""ROCm (gfx950 MFMA) weight layout for Humming's weight-only GEMM.

Humming's standard packed weight is ``[N, K * bits / 32]`` int32: each row is a
little-endian bitstream in which element ``k`` occupies bits ``[k * bits, (k + 1) * bits)``.

The MFMA kernel (``include/humming/kernel/humming_mfma.cuh``) consumes the weight in
chunks of 16 output channels x 128 K values. One chunk is 64 lanes x ``bits`` int32
words; lane ``l`` loads its ``bits`` contiguous words with a single vector load and
feeds them straight into ``dequant<ElementQB, Float16/BFloat16>(words, regs, t)`` for
``t = 0..3``. Call ``t`` yields the 8 B values of MFMA ``t`` (16x16x32), so lane ``l``
holds column ``n = l % 16`` and the values

    k = t * 32 + (l // 16) * 8 + s,   t = 0..3, s = 0..7

Chunks are ordered ``[N / 16][K / 128]`` so a wave streams contiguous memory along K.
The total size is identical to the standard packed weight, so the result is returned
with the same ``[N, K * bits / 32]`` shape. See docs/rocm/mfma_layout.md.
"""

import functools

import torch

MFMA_CHUNK_N = 16
MFMA_CHUNK_K = 128
WAVE_SIZE = 64
VALUES_PER_LANE = MFMA_CHUNK_N * MFMA_CHUNK_K // WAVE_SIZE


def next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def pack_weight(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pure-torch equivalent of ``humming.ops.pack_weight`` for ``[..., K]`` codes."""
    assert values.size(-1) * num_bits % 32 == 0
    codes = values.to(torch.int64) & ((1 << num_bits) - 1)
    bit_offsets = torch.arange(values.size(-1), device=values.device) * num_bits
    num_words = values.size(-1) * num_bits // 32
    words = torch.zeros(*values.shape[:-1], num_words + 1, dtype=torch.int64, device=values.device)
    word_index = bit_offsets // 32
    shift = bit_offsets % 32
    low_part = (codes << shift) & 0xFFFFFFFF
    high_part = codes >> (32 - shift)
    words.scatter_add_(-1, word_index.expand_as(codes), low_part)
    spills = (shift + num_bits > 32).expand_as(codes)
    words.scatter_add_(-1, (word_index + 1).expand_as(codes), torch.where(spills, high_part, 0))
    return _to_int32(words[..., :num_words])


def unpack_weight(packed: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pure-torch equivalent of ``humming.ops.unpack_weight``; returns int32 codes ``[..., K]``."""
    words = packed.to(torch.int64) & 0xFFFFFFFF
    shape_k = packed.size(-1) * 32 // num_bits
    bit_offsets = torch.arange(shape_k, device=packed.device) * num_bits
    word_index = bit_offsets // 32
    shift = bit_offsets % 32
    padded = torch.nn.functional.pad(words, (0, 1))
    low_word = padded.index_select(-1, word_index)
    high_word = padded.index_select(-1, word_index + 1)
    combined = (low_word >> shift) | (high_word << (32 - shift))
    return (combined & ((1 << num_bits) - 1)).to(torch.int32)


def _to_int32(words: torch.Tensor) -> torch.Tensor:
    words = words & 0xFFFFFFFF
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)


@functools.cache
def _padded_word_sources(num_bits: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """For each padded word, the (field, lane value index) pairs Humming's dequant expects.

    ``dequant<ElementQB, 16-bit>(qb, res, t)`` produces register ``i`` (values
    ``2i, 2i + 1`` of MFMA ``t``) from padded word ``(4t + i) * P / 16`` (``P`` = padded
    bits), taking the low half from field ``(4t + i) % (16 / P)`` and the high half from
    that field plus ``16 / P``.
    """
    padded_bits = next_power_of_2(num_bits)
    fields_per_half = 16 // padded_bits
    sources: list[list[tuple[int, int]]] = [[] for _ in range(padded_bits)]
    for mfma_index in range(4):
        for register in range(4):
            index = mfma_index * 4 + register
            word = index * padded_bits // 16
            field = index % fields_per_half
            sources[word].append((field, mfma_index * 8 + register * 2))
            sources[word].append((field + fields_per_half, mfma_index * 8 + register * 2 + 1))
    return tuple(tuple(word_sources) for word_sources in sources)


def _squeeze_padded_words(padded: list[torch.Tensor], num_bits: int) -> list[torch.Tensor]:
    """Compress P padded words into ``num_bits`` words (mirrors humming_pack_weight)."""
    w = padded
    if num_bits == 3:
        return [
            (w[0] & 0x77777777) | ((w[1] & 0x44444444) << 1),
            (w[1] & 0x33333333) | ((w[2] & 0x66666666) << 1),
            (w[2] & 0x11111111) | ((w[3] & 0x77777777) << 1),
        ]
    if num_bits == 5:
        return [
            (w[0] & 0x1F1F1F1F) | ((w[1] & 0x1C1C1C1C) << 3),
            (w[1] & 0x03030303) | ((w[2] & 0x1F1F1F1F) << 2) | ((w[3] & 0x10101010) << 3),
            (w[3] & 0x0F0F0F0F) | ((w[4] & 0x1E1E1E1E) << 3),
            (w[4] & 0x01010101) | ((w[5] & 0x1F1F1F1F) << 1) | ((w[6] & 0x18181818) << 3),
            (w[6] & 0x07070707) | ((w[7] & 0x1F1F1F1F) << 3),
        ]
    if num_bits == 6:
        return [
            (w[0] & 0x3F3F3F3F) | ((w[1] & 0x30303030) << 2),
            (w[1] & 0x0F0F0F0F) | ((w[2] & 0x3C3C3C3C) << 2),
            (w[2] & 0x03030303) | ((w[3] & 0x3F3F3F3F) << 2),
            (w[4] & 0x3F3F3F3F) | ((w[5] & 0x30303030) << 2),
            (w[5] & 0x0F0F0F0F) | ((w[6] & 0x3C3C3C3C) << 2),
            (w[6] & 0x03030303) | ((w[7] & 0x3F3F3F3F) << 2),
        ]
    if num_bits == 7:
        return [
            (w[0] & 0x7F7F7F7F) | ((w[1] & 0x40404040) << 1),
            (w[1] & 0x3F3F3F3F) | ((w[2] & 0x60606060) << 1),
            (w[2] & 0x1F1F1F1F) | ((w[3] & 0x70707070) << 1),
            (w[3] & 0x0F0F0F0F) | ((w[4] & 0x78787878) << 1),
            (w[4] & 0x07070707) | ((w[5] & 0x7C7C7C7C) << 1),
            (w[5] & 0x03030303) | ((w[6] & 0x7E7E7E7E) << 1),
            (w[6] & 0x01010101) | ((w[7] & 0x7F7F7F7F) << 1),
        ]
    return padded


def repack_codes_for_mfma(codes: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Repack unpacked weight codes ``[..., N, K]`` into the MFMA layout ``[..., N, K * bits / 32]``."""
    assert 1 <= num_bits <= 8
    *lead, shape_n, shape_k = codes.shape
    assert shape_n % MFMA_CHUNK_N == 0, "N must be a multiple of 16 (pad first)"
    assert shape_k % MFMA_CHUNK_K == 0, "K must be a multiple of 128 (pad first)"

    # [..., N/16, 16 (lane % 16), K/128, 4 (t), 4 (lane // 16), 8 (s)]
    codes = codes.to(torch.int64) & ((1 << num_bits) - 1)
    codes = codes.reshape(*lead, shape_n // 16, 16, shape_k // 128, 4, 4, 8)
    ndim = len(lead)
    # -> [..., N/16, K/128, lane // 16, lane % 16, t, s] = [..., N/16, K/128, 64 lanes, 32 values]
    order = (*range(ndim), ndim, ndim + 2, ndim + 4, ndim + 1, ndim + 3, ndim + 5)
    lane_values = codes.permute(*order).reshape(
        *lead, shape_n // 16, shape_k // 128, WAVE_SIZE, VALUES_PER_LANE
    )

    padded_bits = next_power_of_2(num_bits)
    padded_words = []
    for word_sources in _padded_word_sources(num_bits):
        word = torch.zeros(lane_values.shape[:-1], dtype=torch.int64, device=codes.device)
        for field, value_index in word_sources:
            word |= lane_values[..., value_index] << (field * padded_bits)
        padded_words.append(word)

    words = torch.stack(_squeeze_padded_words(padded_words, num_bits), dim=-1)
    return _to_int32(words).reshape(*lead, shape_n, shape_k * num_bits // 32)


def repack_weight_for_mfma(packed_weight: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Convert Humming's standard packed weight ``[..., N, K * bits / 32]`` to the MFMA layout."""
    return repack_codes_for_mfma(unpack_weight(packed_weight, num_bits), num_bits)


def transform_weight_scale(weight_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[..., N, num_groups]`` scales -> ``[..., num_groups, N]`` in the activation dtype."""
    return weight_scale.to(dtype).transpose(-1, -2).contiguous()


def transform_zero_point(zero_point: torch.Tensor) -> torch.Tensor:
    """Unpacked integer zero points ``[..., N, num_groups]`` -> ``[..., num_groups, N]`` uint8."""
    return zero_point.to(torch.uint8).transpose(-1, -2).contiguous()


def dequantize_reference(
    codes: torch.Tensor,
    weight_scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    num_bits: int,
    group_size: int,
) -> torch.Tensor:
    """FP32 reference: ``(code - zp) * scale`` with ``zp = 2^(bits-1)`` when not given."""
    shape_n, shape_k = codes.shape[-2:]
    group_size = group_size or shape_k
    values = codes.float()
    if zero_point is None:
        values = values - (1 << (num_bits - 1))
    else:
        values = values - zero_point.float().repeat_interleave(group_size, dim=-1)[..., :shape_k]
    return values * weight_scale.float().repeat_interleave(group_size, dim=-1)[..., :shape_k]

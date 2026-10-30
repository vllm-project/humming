"""ROCm (gfx950) weight layout and lookup tables for the low-bit activation GEMM.

The kernel (``include/humming/kernel/humming_mfma_f8.cuh``) consumes B in chunks of 16 output
channels x 128 K values. One chunk is 64 lanes x ``bits`` int32 words, ordered
``[N / 16][K / 128][64 lanes][bits words]`` (the same total size, and the same ``[N, K * bits /
32]`` shape, as Humming's standard packed weight). Lane ``l`` holds column ``l % 16`` and 32
values; value ``j`` is the K index given by the MFMA operand layout (``lg = l // 16``):

    "L128": k = (j // 16) * 64 + lg * 16 + j % 16   8-bit operand of the K = 128 scaled MFMA,
                                                     also two INT8 16x16x64 MFMAs
    "L32":  k = (j // 8) * 32 + lg * 8 + j % 8        four K = 32 FP8 / INT8 MFMAs
    "TRUE": k = lg * 32 + j                          FP4 / FP6 operands of the K = 128 MFMA

Packing of the 32 values of a lane:

    "stream": little-endian bitstream (value j at bits [j * bits, (j + 1) * bits)); this is the
              raw MFMA operand for native FP8 / FP6 / FP4 weights.
    "planes": codes dequantized in registers (table lookup / integer arithmetic) are split into
              bit planes of 4 / 2 / 1 bits (low bits first). A plane of width w takes w words; in
              them value j = 4 r + b (register r, byte b) sits at bit 8 b + w * (r % (8 / w)) of
              word r // (8 / w), so one shift and one mask yield 4 byte-aligned codes
              (``MfmaF8Planes`` in the kernel).
"""

import functools

import torch

from humming.rocm.layout import pack_weight, unpack_weight

CHUNK_N = 16
CHUNK_K = 128
WAVE_SIZE = 64
VALUES_PER_LANE = CHUNK_N * CHUNK_K // WAVE_SIZE
LAYOUTS = ("L128", "L32", "TRUE")


@functools.cache
def _lane_k_index_cpu(layout: str) -> torch.Tensor:
    lane = torch.arange(WAVE_SIZE).view(-1, 1)
    j = torch.arange(VALUES_PER_LANE).view(1, -1)
    lg = lane // 16
    if layout == "L128":
        k = (j // 16) * 64 + lg * 16 + j % 16
    elif layout == "L32":
        k = (j // 8) * 32 + lg * 8 + j % 8
    elif layout == "TRUE":
        k = lg * 32 + j
    else:
        raise ValueError(f"unknown MFMA operand layout {layout}")
    return k


def lane_k_index(layout: str, device=None) -> torch.Tensor:
    """``[64, 32]`` int64: K offset (within a 128 chunk) of value j of lane l."""
    return _lane_k_index_cpu(layout).to(device)


def plane_widths(num_bits: int) -> list[int]:
    if num_bits == 8:
        return [8]
    widths = []
    for width in (4, 2, 1):
        if num_bits & width:
            widths.append(width)
    return widths


def _pack_planes(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    """``[..., 32]`` codes -> ``[..., num_bits]`` int32 words in the bit-plane packing."""
    values = values.to(torch.int64)
    words = []
    bit_offset = 0
    for width in plane_widths(num_bits):
        per_word = 8 // width
        plane = (values >> bit_offset) & ((1 << width) - 1)
        for word_index in range(width):
            word = torch.zeros(values.shape[:-1], dtype=torch.int64, device=values.device)
            for reg in range(word_index * per_word, (word_index + 1) * per_word):
                for byte in range(4):
                    word |= plane[..., 4 * reg + byte] << (8 * byte + width * (reg % per_word))
            words.append(word)
        bit_offset += width
    words = torch.stack(words, dim=-1) & 0xFFFFFFFF
    return torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32)


def _unpack_planes(words: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Inverse of ``_pack_planes`` (tests): ``[..., num_bits]`` words -> ``[..., 32]`` codes."""
    words = words.to(torch.int64) & 0xFFFFFFFF
    codes = torch.zeros(*words.shape[:-1], VALUES_PER_LANE, dtype=torch.int64, device=words.device)
    word_base = 0
    bit_offset = 0
    for width in plane_widths(num_bits):
        per_word = 8 // width
        for reg in range(8):
            word = words[..., word_base + reg // per_word]
            for byte in range(4):
                plane = (word >> (8 * byte + width * (reg % per_word))) & ((1 << width) - 1)
                codes[..., 4 * reg + byte] |= plane << bit_offset
        word_base += width
        bit_offset += width
    return codes.to(torch.int32)


def repack_codes(codes: torch.Tensor, num_bits: int, layout: str, packing: str) -> torch.Tensor:
    """Unpacked codes ``[N, K]`` -> the kernel's B layout ``[N, K * bits / 32]`` int32."""
    assert packing in ("stream", "planes")
    assert packing == "stream" or num_bits in (1, 2, 3, 4, 5, 6, 7, 8)
    shape_n, shape_k = codes.shape
    assert shape_n % CHUNK_N == 0 and shape_k % CHUNK_K == 0
    k_index = lane_k_index(layout, codes.device)
    lane_row = torch.arange(WAVE_SIZE, device=codes.device).view(-1, 1) % 16
    # [N/16, 16, K/128, 128] -> [N/16, K/128, 16, 128] -> lane values [N/16, K/128, 64, 32]
    chunks = codes.reshape(shape_n // CHUNK_N, CHUNK_N, shape_k // CHUNK_K, CHUNK_K).permute(0, 2, 1, 3)
    values = chunks[:, :, lane_row, k_index]
    if packing == "stream":
        words = pack_weight(values, num_bits)
    else:
        words = _pack_planes(values, num_bits)
    return words.reshape(shape_n, shape_k * num_bits // 32).contiguous()


def repack_weight(packed_weight: torch.Tensor, num_bits: int, layout: str, packing: str) -> torch.Tensor:
    """Humming's standard packed weight ``[N, K * bits / 32]`` -> the kernel's B layout."""
    shape_k = packed_weight.size(1) * 32 // num_bits
    # Bound the int64 temporaries of the pure-torch repack to ~2^25 codes per slice.
    rows = max(CHUNK_N, (1 << 25) // shape_k // CHUNK_N * CHUNK_N)
    parts = [
        repack_codes(unpack_weight(part, num_bits), num_bits, layout, packing)
        for part in packed_weight.split(rows, dim=0)
    ]
    return torch.cat(parts, dim=0).contiguous()


def unrepack_weight(weight: torch.Tensor, num_bits: int, layout: str, packing: str) -> torch.Tensor:
    """Inverse of ``repack_codes`` (tests): kernel layout -> codes ``[N, K]``."""
    shape_n = weight.size(0)
    shape_k = weight.size(1) * 32 // num_bits
    words = weight.reshape(shape_n // CHUNK_N, shape_k // CHUNK_K, WAVE_SIZE, num_bits)
    if packing == "stream":
        values = unpack_weight(words, num_bits)
    else:
        values = _unpack_planes(words, num_bits)
    k_index = lane_k_index(layout, weight.device)
    lane_row = torch.arange(WAVE_SIZE, device=weight.device).view(-1, 1).expand(-1, VALUES_PER_LANE) % 16
    codes = torch.empty(
        shape_n // CHUNK_N, shape_k // CHUNK_K, CHUNK_N, CHUNK_K, dtype=torch.int32, device=weight.device
    )
    codes[:, :, lane_row, k_index.expand(WAVE_SIZE, -1)] = values
    return codes.permute(0, 2, 1, 3).reshape(shape_n, shape_k)


# ----------------------------------------------------------------------------------------------
# Lookup tables: code (or code + zero-point offset) -> FP8 e4m3 / INT8 byte.


def _fp8_e4m3_bytes(values: torch.Tensor) -> torch.Tensor | None:
    """FP8 e4m3 bytes of ``values`` or None if any value is not exactly representable."""
    values = values.float().cpu()
    fp8 = values.to(torch.float8_e4m3fn)
    if not torch.equal(fp8.float(), values):
        return None
    return fp8.view(torch.uint8).to(torch.int64)


def _int8_bytes(values: torch.Tensor) -> torch.Tensor | None:
    values = values.float().cpu()
    if not torch.equal(values.round(), values) or values.min() < -128 or values.max() > 127:
        return None
    return values.to(torch.int64) & 0xFF


def table_bytes(values: torch.Tensor, target: str) -> torch.Tensor | None:
    """Byte encodings of the table values in ``target`` ("fp8" e4m3 or "int8"), or None."""
    return _fp8_e4m3_bytes(values) if target == "fp8" else _int8_bytes(values)


def table_words(encoded: torch.Tensor) -> tuple[int, ...]:
    """Up to 32 table bytes -> 8 words (entry e = byte e % 4 of word e // 4), zero padded."""
    assert encoded.numel() <= 32
    padded = torch.zeros(32, dtype=torch.int64)
    padded[: encoded.numel()] = encoded
    words = padded.view(8, 4)
    return tuple(int(w[0] | (w[1] << 8) | (w[2] << 16) | (w[3] << 24)) for w in words)


def int_table_values(num_bits: int, has_zero_point: bool) -> torch.Tensor:
    """Values indexed by the kernel's LUT index for integer codes.

    Symmetric: index = code, value = code - 2^(bits-1).
    Zero point: index = code + (2^bits - 1 - zp), value = index - (2^bits - 1) = code - zp.
    """
    if has_zero_point:
        return torch.arange(2 ** (num_bits + 1) - 1, dtype=torch.float32) - (2**num_bits - 1)
    return torch.arange(2**num_bits, dtype=torch.float32) - 2 ** (num_bits - 1)


def fp_table_values(exponent_bits: int, mantissa_bits: int, num_bits: int) -> torch.Tensor:
    """Values of every code of a (signed) Humming FP format, as the reference dequantizes them."""
    from humming import ops

    codes = torch.arange(2**num_bits, dtype=torch.int32, device="cuda")
    return ops.dequant_weight(codes, exponent_bits, mantissa_bits, True).float().cpu()


# ----------------------------------------------------------------------------------------------
# Scales / zero points.


def group_scale_fp32(weight_scale: torch.Tensor) -> torch.Tensor:
    """``[N, G]`` group scales (any float dtype) -> ``[G, N]`` FP32."""
    return weight_scale.float().transpose(0, 1).contiguous()


def group_scale_e8m0(weight_scale: torch.Tensor) -> torch.Tensor:
    """``[N, G]`` E8M0 scales -> ``[G, N]`` uint8 exponent bytes."""
    return weight_scale.contiguous().view(torch.uint8).transpose(0, 1).contiguous()


def block_scale_fp32(weight_scale: torch.Tensor, group_size_n: int, shape_n: int) -> torch.Tensor:
    """``[N / gn, G]`` block scales -> ``[G, N]`` FP32 (every column gets its block's scale)."""
    expanded = weight_scale.float().repeat_interleave(group_size_n, dim=0)[:shape_n]
    return expanded.transpose(0, 1).contiguous()


def zero_point_codes(zero_point: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Packed-along-N zero points ``[N * bits / 32, G]`` -> ``[G, N]`` uint8 codes."""
    codes = unpack_weight(zero_point.transpose(0, 1).contiguous(), num_bits)
    return codes.to(torch.uint8).contiguous()

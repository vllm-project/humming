#pragma once
// ROCm replacement for utils/ptx/math.cuh (prmt / lop3).

// PTX prmt in default mode: each selector nibble picks one byte of the 8-byte value {b, a}
// (a = bytes 0-3, b = bytes 4-7); a set bit 3 replicates that byte's sign bit.
// v_perm_b32(x, y, sel) picks from {x, y} with y = bytes 0-3, so prmt(a, b) == perm(b, a, sel')
// where sel' holds the low three bits of each prmt nibble in its byte lanes.
CUDA_INLINE uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  const bool uses_sign_replication = (s & 0x8888u) != 0;
  if (!uses_sign_replication) {
    const uint32_t byte_selector =
        (s & 0x7u) | (((s >> 4) & 0x7u) << 8) | (((s >> 8) & 0x7u) << 16) | (((s >> 12) & 0x7u) << 24);
    return __builtin_amdgcn_perm(b, a, byte_selector);
  }
  const uint64_t source_bytes = (static_cast<uint64_t>(b) << 32) | a;
  uint32_t result = 0;
  PRAGMA_UNROLL
  for (uint32_t output_byte = 0; output_byte < 4; output_byte++) {
    const uint32_t nibble = (s >> (4 * output_byte)) & 0xFu;
    const uint32_t source_byte = static_cast<uint32_t>(source_bytes >> (8 * (nibble & 0x7u))) & 0xFFu;
    const bool replicate_sign = (nibble & 0x8u) != 0;
    const uint32_t selected_byte = replicate_sign ? ((source_byte & 0x80u) ? 0xFFu : 0x00u) : source_byte;
    result |= selected_byte << (8 * output_byte);
  }
  return result;
};

// lop3: output bit = lut[(a << 2) | (b << 1) | c] per bit position.
// gfx950 has a native LOP3, v_bitop3_b32, with the same truth-table encoding as PTX lop3
// (checked on MI355X, see docs/rocm/log_mfma.md iteration 5). It matters for the dequant
// hot loop: gfx9 VOP3 cannot encode literals, so the generic (a & mask) | magic costs two
// ops plus a constant register. Other targets expand the LUT's minterms; with a constant
// LUT the compiler folds that to a few v_bfi/v_and_or/v_xor ops.
template <uint32_t lut>
CUDA_INLINE uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  static_assert(lut <= 0xFF, "lop3 lut is 8 bits");
#if defined(__gfx950__)
  return __builtin_amdgcn_bitop3_b32(a, b, c, lut);
#else
  uint32_t result = 0;
  static_for<0, 8>([&](auto minterm_index) {
    constexpr uint32_t minterm = decltype(minterm_index)::value;
    if constexpr ((lut >> minterm) & 1u) {
      const uint32_t a_term = (minterm & 4u) ? a : ~a;
      const uint32_t b_term = (minterm & 2u) ? b : ~b;
      const uint32_t c_term = (minterm & 1u) ? c : ~c;
      result |= a_term & b_term & c_term;
    }
  });
  return result;
#endif
};

CUDA_INLINE uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) { return lop3<(0xF0 & 0xCC) | 0xAA>(a, b, c); };

CUDA_INLINE uint32_t lop3_and_xor(uint32_t a, uint32_t b, uint32_t c) { return lop3<(0xF0 & 0xCC) ^ 0xAA>(a, b, c); };

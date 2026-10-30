#pragma once

// gfx950 (CDNA4) 8/6/4-bit MFMA wrappers and byte-wise dequant helpers for the low-bit
// activation GEMM (humming/kernel/humming_mfma_f8.cuh).
//
// Fragment layouts (wave64, verified on MI355X by tests/rocm/test_mfma_f8_layout.py, see
// docs/rocm/log_f8.md). lg = lane / 16; every lane holds row / column lane % 16:
//
//   v_mfma_scale_f32_16x16x128_f8f6f4 (format codes 0 e4m3, 1 e5m2, 2 e2m3, 3 e3m2, 4 e2m1)
//     8-bit operand (8 VGPRs), byte j:     k = (j / 16) * 64 + lg * 16 + j % 16   ("L128")
//     6-bit operand (6 VGPRs), bits 6j:    k = lg * 32 + j                        ("TRUE")
//     4-bit operand (4 VGPRs), nibble j:   k = lg * 32 + j                        ("TRUE")
//     E8M0 scale operand (opsel 0, byte 0): lane l scales row / column l % 16 for the K
//     block [32 * lg, 32 * lg + 32); 127 = 1.0.
//   v_mfma_i32_16x16x64_i8 (4 VGPRs), byte s:    k = lg * 16 + s  (two of them = L128)
//   v_mfma_{i32,f32}_16x16x32_{i8,fp8,bf8} (2 VGPRs), byte s:  k = lg * 8 + s   ("L32")
//   C (4 regs): column lane % 16, row (lane / 16) * 4 + r.
//
// Throughput on MI355X (one wave, dependent chains, relative to f16 16x16x32): 16x16x128
// f8f6f4 with an 8-bit operand 2x (= 2x the f16 FLOP rate), f4/f6 x f4/f6 1x (4x), i8 x64
// 1x, fp8/i8 x32 1x. The K = 32 FP8 MFMA is no faster than FP16, so FP8 activations use the
// K = 128 scaled MFMA whenever the scale granularity allows it.

#include <humming/mma/mfma.cuh>

#ifdef HUMMING_HIP

using mfma_i32x8_t = int32_t __attribute__((ext_vector_type(8)));
using mfma_i32x4_t = int32_t __attribute__((ext_vector_type(4)));

// MFMA operand format codes (cbsz / blgp of the f8f6f4 MFMA). kMfmaFmtInt8 is ours.
constexpr uint32_t kMfmaFmtE4M3 = 0;
constexpr uint32_t kMfmaFmtE5M2 = 1;
constexpr uint32_t kMfmaFmtE2M3 = 2;
constexpr uint32_t kMfmaFmtE3M2 = 3;
constexpr uint32_t kMfmaFmtE2M1 = 4;
constexpr uint32_t kMfmaFmtInt8 = 8;

CUDA_INLINE constexpr uint32_t mfma_fmt_regs(uint32_t fmt) {
  return fmt == kMfmaFmtE2M1 ? 4 : (fmt == kMfmaFmtE2M3 || fmt == kMfmaFmtE3M2) ? 6 : 8;
}

// D += A x B over K = 128 with E8M0 block scales (scale_a / scale_b: byte 0 used).
template <uint32_t kFmtA, uint32_t kFmtB>
CUDA_INLINE void mfma_scale_16x16x128(const uint32_t *a, const uint32_t *b, mfma_f32x4_t &c, uint32_t scale_a, uint32_t scale_b) {
  mfma_i32x8_t a_vec, b_vec;
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 8; i++) {
    a_vec[i] = i < mfma_fmt_regs(kFmtA) ? static_cast<int32_t>(a[i]) : 0;
    b_vec[i] = i < mfma_fmt_regs(kFmtB) ? static_cast<int32_t>(b[i]) : 0;
  }
  c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      a_vec, b_vec, c, kFmtA, kFmtB, 0, static_cast<int32_t>(scale_a), 0, static_cast<int32_t>(scale_b));
}

// D += A x B over K = 64, signed int8 (4 regs each).
CUDA_INLINE void mfma_i8_16x16x64(const uint32_t *a, const uint32_t *b, mfma_i32x4_t &c) {
  const mfma_i32x4_t a_vec = {static_cast<int32_t>(a[0]), static_cast<int32_t>(a[1]), static_cast<int32_t>(a[2]), static_cast<int32_t>(a[3])};
  const mfma_i32x4_t b_vec = {static_cast<int32_t>(b[0]), static_cast<int32_t>(b[1]), static_cast<int32_t>(b[2]), static_cast<int32_t>(b[3])};
  c = __builtin_amdgcn_mfma_i32_16x16x64_i8(a_vec, b_vec, c, 0, 0, 0);
}

CUDA_INLINE long mfma_pack_b64(const uint32_t *regs) {
  return static_cast<long>((static_cast<uint64_t>(regs[1]) << 32) | regs[0]);
}

// D += A x B over K = 32 (2 regs each): int8 x int8, or fp8 / bf8 x fp8 / bf8.
CUDA_INLINE void mfma_i8_16x16x32(const uint32_t *a, const uint32_t *b, mfma_i32x4_t &c) {
  c = __builtin_amdgcn_mfma_i32_16x16x32_i8(mfma_pack_b64(a), mfma_pack_b64(b), c, 0, 0, 0);
}

template <uint32_t kFmtA, uint32_t kFmtB>
CUDA_INLINE void mfma_f8_16x16x32(const uint32_t *a, const uint32_t *b, mfma_f32x4_t &c) {
  static_assert((kFmtA == kMfmaFmtE4M3 || kFmtA == kMfmaFmtE5M2) && (kFmtB == kMfmaFmtE4M3 || kFmtB == kMfmaFmtE5M2));
  const long a_vec = mfma_pack_b64(a);
  const long b_vec = mfma_pack_b64(b);
  if constexpr (kFmtA == kMfmaFmtE4M3 && kFmtB == kMfmaFmtE4M3) {
    c = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_vec, b_vec, c, 0, 0, 0);
  } else if constexpr (kFmtA == kMfmaFmtE4M3) {
    c = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a_vec, b_vec, c, 0, 0, 0);
  } else if constexpr (kFmtB == kMfmaFmtE4M3) {
    c = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a_vec, b_vec, c, 0, 0, 0);
  } else {
    c = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a_vec, b_vec, c, 0, 0, 0);
  }
}


// Byte-wise table lookup: byte i of the result is table[x.byte(i)] for indices < 2^kIndexBits.
// `table` holds 4 entries per word (entry e = byte e % 4 of word e / 4). v_perm_b32 selects one of
// 8 bytes per selector byte, so every 8 entries cost one v_perm; the halves are merged with
// byte masks built by v_perm itself (selector 12 -> 0x00, >= 13 -> 0xFF).
//   kIndexBits 3: 1 op, 4: 6 ops, 5: 12 ops per 4 values.
template <uint32_t kIndexBits>
CUDA_INLINE uint32_t mfma_lut_lookup(uint32_t x, const uint32_t *table) {
  static_assert(kIndexBits >= 1 && kIndexBits <= 5);
  if constexpr (kIndexBits <= 3) {
    return __builtin_amdgcn_perm(table[1], table[0], x);
  } else {
    const uint32_t index = x & 0x07070707u;
    // Selector 0x05 -> byte 1 of the zero operand, 0x0D -> 0xFF.
    const uint32_t mask3 = __builtin_amdgcn_perm(0u, 0u, (x & 0x08080808u) | 0x05050505u);
    const uint32_t lo = __builtin_amdgcn_perm(table[1], table[0], index);
    const uint32_t hi = __builtin_amdgcn_perm(table[3], table[2], index);
    const uint32_t first = (hi & mask3) | (lo & ~mask3);
    if constexpr (kIndexBits == 4) {
      return first;
    } else {
      // Selector 0x0C -> 0x00, 0x1C -> 0xFF.
      const uint32_t mask4 = __builtin_amdgcn_perm(0u, 0u, (x & 0x10101010u) | 0x0C0C0C0Cu);
      const uint32_t lo2 = __builtin_amdgcn_perm(table[5], table[4], index);
      const uint32_t hi2 = __builtin_amdgcn_perm(table[7], table[6], index);
      const uint32_t second = (hi2 & mask3) | (lo2 & ~mask3);
      return (second & mask4) | (first & ~mask4);
    }
  }
}

CUDA_INLINE float mfma_e8m0_to_float(uint32_t e) {
  e &= 0xFFu;
  return __uint_as_float(e == 0 ? 0x00400000u : e << 23);
}

#endif

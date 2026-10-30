#pragma once

// Weight-only quantized GEMM for AMD CDNA (gfx950), built on MFMA 16x16x32 FP16/BF16.
//
//   C[M, N] = A[M, K] @ dequant(B)[N, K]^T
//
// A / C: row-major FP16 or BF16. B: unsigned integer codes (1..8 bits) in the ROCm MFMA
// layout produced by humming/rocm/layout.py: chunks of 16 channels x 128 K, each chunk is
// 64 lanes x kBits int32 words, chunks ordered [N / 16][K / 128]. Lane l of a wave loads
// its kBits words with one vector load and Humming's dequant<ElementQB, ElementA>(qb, res, t)
// turns them directly into the B fragment of MFMA t (t = 0..3), no shuffles needed.
//
// Scales: [num_groups, N] in the activation dtype. Group scales are applied to the FP32
// accumulator (every lane's C fragment is a single column, so this is one FMA per
// accumulator register per group), per-channel scales in the epilogue.
// Zero points (optional): [num_groups, N] uint8; without them the code is centered at
// 2^(bits - 1) (symmetric quantization).
//
// Block = kWarpsM x kWarpsN waves; each wave owns kWarpTilesM x kWarpTilesN 16x16 output
// tiles and streams its own B directly from global memory into registers with a
// kStages-deep register prefetch. gridDim.z > 1 splits K; the last block of a tile
// reduces the FP32 partials in a fixed order (deterministic) and resets the tile counter.

#include <humming/datatype/dequant.cuh>
#include <humming/mma/mfma.cuh>
#include <humming/utils/all.cuh>


// kLdsA: stage the block's A tile through LDS (shared by all waves along N) instead of
// every wave loading its own A fragments from global memory; needed once M is large.
// kScaleB: apply group scales to the dequantized 16-bit B fragment (packed multiply, as
// Humming's CUDA mainloop does) instead of to a per-group FP32 accumulator. Saves the
// temporary accumulators and the MFMA -> VALU dependency at every group boundary, at the
// cost of rounding (code - zp) * scale to FP16/BF16.
template <
    uint32_t kWarpsM_, uint32_t kWarpsN_, uint32_t kWarpTilesM_, uint32_t kWarpTilesN_, uint32_t kStages_,
    bool kLdsA_ = false, bool kScaleB_ = false>
struct MfmaTileConfig {
  static constexpr uint32_t kWarpsM = kWarpsM_;
  static constexpr uint32_t kWarpsN = kWarpsN_;
  static constexpr uint32_t kWarpTilesM = kWarpTilesM_;
  static constexpr uint32_t kWarpTilesN = kWarpTilesN_;
  static constexpr uint32_t kStages = kStages_;
  static constexpr uint32_t kNumThreads = kWarpsM * kWarpsN * kWaveSize;
  static constexpr uint32_t kBlockM = kWarpsM * kWarpTilesM * 16;
  static constexpr uint32_t kBlockN = kWarpsN * kWarpTilesN * 16;
  static constexpr bool kLdsA = kLdsA_;
  static constexpr bool kScaleB = kScaleB_;
  // 16-byte A units (8 values) per thread per 128-wide K chunk when staging through LDS.
  static constexpr uint32_t kLdsAUnitsPerThread = kLdsA ? kBlockM * 16 / kNumThreads : 0;
  // One LDS A buffer per pipeline stage: chunk loads go straight to LDS (buffer_load ... lds).
  static constexpr uint32_t kLdsAUnits = kLdsA ? kStages * kBlockM * 16 : 1;
  static_assert(kStages >= 2, "register prefetch needs at least two stages");
  static_assert(!kLdsA || kBlockM * 16 % kNumThreads == 0);
};


template <class ElementA>
CUDA_INLINE float mfma_half_to_float(uint16_t bits) {
  if constexpr (std::is_same<ElementA, Float16>::value) {
    return __half2float(__ushort_as_half(bits));
  } else {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
  }
}

template <class ElementA>
CUDA_INLINE uint16_t mfma_float_to_half(float value) {
  if constexpr (std::is_same<ElementA, Float16>::value) {
    return __half_as_ushort(__float2half_rn(value));
  } else {
    // Round to nearest even; NaN inputs are not expected from finite GEMMs.
    uint32_t bits = __float_as_uint(value);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
  }
}


// Dequantize the lane's packed words into the 16-bit B fragment of MFMA `mfma_index`.
//
// Both activation types use Humming's FP16 magic-number dequant: (code - zero point) is an
// exact FP16 integer for codes up to 10 bits and costs packed FP16 ops. gfx950 has no
// packed BF16 arithmetic (Humming's BF16 path would convert every __hsub2 through FP32),
// so for BF16 the exact FP16 pair is widened with v_cvt_f32_f16 / v_fma_mix_f32, which
// can also fold in the group scale, and narrowed with one v_cvt_pk_bf16_f32.
template <class ElementQB, class ElementA, bool kHasZeroPoint>
struct MfmaDequant {
  static constexpr uint32_t kBits = ElementQB::kBits;
  static constexpr bool kIsBFloat16 = std::is_same<ElementA, BFloat16>::value;
  static_assert(ElementQB::kIsIntegerType && !ElementQB::kIsSigned);
  static_assert(kBits <= Float16::kMantissaBits - kHasZeroPoint);

  using Half2 = _Float16 __attribute__((ext_vector_type(2)));
  using BFloat2 = __bf16 __attribute__((ext_vector_type(2)));
  using Float2 = float __attribute__((ext_vector_type(2)));

  // Shift-light path for 1/2/4/8-bit codes (same register assignment as Humming's dequant_b1248,
  // bit-identical results). A field at bit offset r (r + kBits <= 10) of a 16-bit half is
  // decoded in place with magic 2^(10 - r): (w & (mask << r)) | bits(2^(10 - r)) is the FP16
  // value 2^(10 - r) + code. So each word needs at most one shift (by 8) shared by the fields
  // in its upper byte, instead of one shift per field: int4 drops from 3 to 1 v_lshrrev per
  // fragment (15 -> 13 VALU with the group scale).
  static constexpr bool kShiftLight = kBits == 1 || kBits == 2 || kBits == 4 || kBits == 8;

  CUDA_INLINE static constexpr uint32_t magic_bits(uint32_t residual) {
    return (0x6400u - residual * 0x400u) * 0x00010001u;
  }

  // `zero_point` is the raw code; the result is what run() expects as `prepared_zero_point`.
  CUDA_INLINE static uint32_t prepare_zero_point(uint32_t zero_point) {
    if constexpr (kHasZeroPoint && kShiftLight) {
      return (zero_point & ((1u << kBits) - 1)) * 0x00010001u;
    } else if constexpr (kHasZeroPoint) {
      return dequant_single_zero_point<ElementQB, Float16>(zero_point);
    } else {
      return 0;
    }
  }

  CUDA_INLINE static void dequant_f16(const uint32_t *qb, uint32_t *regs_b, uint32_t mfma_index, uint32_t prepared_zero_point) {
    if constexpr (kShiftLight) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        // Field position of register i, exactly as in dequant_b1248.
        const uint32_t index = mfma_index * 4 + i;
        const uint32_t word = index * 2 * kBits / 32;
        const uint32_t shift = (kBits * 4 % 16 == 0 ? kBits * i : kBits * index) % 16;
        const uint32_t residual = shift % 8;
        const uint32_t mask = ((1u << kBits) - 1) * 0x00010001u << residual;
        const uint32_t value = lop3_and_or(qb[word] >> (shift - residual), mask, magic_bits(residual));
        uint32_t offset;
        if constexpr (kHasZeroPoint) {
          offset = (prepared_zero_point << residual) + magic_bits(residual);
        } else {
          offset = (0x00010001u << (kBits - 1) << residual) + magic_bits(residual);
        }
        const Half2 result = __builtin_bit_cast(Half2, value) - __builtin_bit_cast(Half2, offset);
        regs_b[i] = __builtin_bit_cast(uint32_t, result);
      }
    } else {
      uint32_t zp_vals[4] = {prepared_zero_point, prepared_zero_point, prepared_zero_point, prepared_zero_point};
      dequant<ElementQB, Float16, kHasZeroPoint, false>(qb, regs_b, mfma_index, zp_vals);
    }
  }

  CUDA_INLINE static uint32_t f16x2_to_bf16x2(uint32_t value, float scale) {
    const Half2 half2 = __builtin_bit_cast(Half2, value);
    const Float2 float2 = {static_cast<float>(half2.x) * scale, static_cast<float>(half2.y) * scale};
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(float2, BFloat2));
  }

  // BF16 shift-light path: the FP16 field x = 2^(10 - r) + code goes straight into
  // v_fma_mix_f32(x, s, -(2^(10 - r) + zp) * s). Both products fit in 19 bits, so the FMA
  // yields (code - zp) * s exactly in FP32 and v_cvt_pk_bf16_f32 rounds once: bit-identical
  // to the FP16-subtract path, at 4 VALU per register instead of 6 (no packed BF16 math on gfx950).
  CUDA_INLINE static void dequant_bf16_scaled(const uint32_t *qb, uint32_t *regs_b, uint32_t mfma_index, uint32_t prepared_zero_point, float scale) {
    const float zero_point = kHasZeroPoint ? static_cast<float>(prepared_zero_point & 0xFFFFu) : static_cast<float>(1u << (kBits - 1));
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      const uint32_t index = mfma_index * 4 + i;
      const uint32_t word = index * 2 * kBits / 32;
      const uint32_t shift = (kBits * 4 % 16 == 0 ? kBits * i : kBits * index) % 16;
      const uint32_t residual = shift % 8;
      const uint32_t mask = ((1u << kBits) - 1) * 0x00010001u << residual;
      const uint32_t value = lop3_and_or(qb[word] >> (shift - residual), mask, magic_bits(residual));
      const float offset = -(static_cast<float>(1u << (10 - residual)) + zero_point) * scale;
      Float2 float2;
#if defined(__HIP_DEVICE_COMPILE__)
      // LLVM does not form v_fma_mix_f32 here (it emits cvt_f32_f16 + fma), so spell it out.
      asm("v_fma_mix_f32 %0, %1, %2, %3 op_sel_hi:[1,0,0]" : "=v"(float2.x) : "v"(value), "v"(scale), "v"(offset));
      asm("v_fma_mix_f32 %0, %1, %2, %3 op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "=v"(float2.y) : "v"(value), "v"(scale), "v"(offset));
#else
      const Half2 half2 = __builtin_bit_cast(Half2, value);
      float2 = {static_cast<float>(half2.x) * scale + offset, static_cast<float>(half2.y) * scale + offset};
#endif
      regs_b[i] = __builtin_bit_cast(uint32_t, __builtin_convertvector(float2, BFloat2));
    }
  }

  // Unscaled (code - zero point) in ElementA.
  CUDA_INLINE static void run(const uint32_t *qb, uint32_t *regs_b, uint32_t mfma_index, uint32_t prepared_zero_point) {
    if constexpr (kIsBFloat16 && kShiftLight) {
      dequant_bf16_scaled(qb, regs_b, mfma_index, prepared_zero_point, 1.0f);
      return;
    }
    dequant_f16(qb, regs_b, mfma_index, prepared_zero_point);
    if constexpr (kIsBFloat16) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) regs_b[i] = f16x2_to_bf16x2(regs_b[i], 1.0f);
    }
  }

  // (code - zero point) * scale in ElementA; `scale` is the raw 16-bit ElementA value.
  CUDA_INLINE static void run_scaled(const uint32_t *qb, uint32_t *regs_b, uint32_t mfma_index, uint32_t prepared_zero_point, uint16_t scale) {
    if constexpr (kIsBFloat16 && kShiftLight) {
      dequant_bf16_scaled(qb, regs_b, mfma_index, prepared_zero_point, __uint_as_float(static_cast<uint32_t>(scale) << 16));
      return;
    }
    dequant_f16(qb, regs_b, mfma_index, prepared_zero_point);
    if constexpr (kIsBFloat16) {
      const float scale_f32 = __uint_as_float(static_cast<uint32_t>(scale) << 16);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) regs_b[i] = f16x2_to_bf16x2(regs_b[i], scale_f32);
    } else {
      const Half2 scale2 = __builtin_bit_cast(Half2, static_cast<uint32_t>(scale) * 0x00010001u);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) regs_b[i] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(Half2, regs_b[i]) * scale2);
    }
  }
};


template <class ElementA, class ElementQB, class TileConfig, uint32_t kGroupSize, bool kHasZeroPoint>
struct MfmaWeightOnlyGemm {
  using Op = MfmaOp<ElementA, 16>;
  using Dequant = MfmaDequant<ElementQB, ElementA, kHasZeroPoint>;
  using CRegisters = typename Op::CRegisters;

  static constexpr uint32_t kBits = ElementQB::kBits;
  static constexpr uint32_t kTilesM = TileConfig::kWarpTilesM;
  static constexpr uint32_t kTilesN = TileConfig::kWarpTilesN;
  static constexpr uint32_t kStages = TileConfig::kStages;
  static constexpr uint32_t kChunkK = 128;
  static constexpr uint32_t kMfmaPerChunk = kChunkK / Op::K;
  static constexpr bool kIsChannelScale = kGroupSize == 0;
  static_assert(kIsChannelScale || kGroupSize % Op::K == 0, "group size must be a multiple of 32");
  // Scales (and zero points) referenced by one 128-wide K chunk.
  static constexpr uint32_t kGroupsPerChunk = kIsChannelScale ? 1 : MAX(kChunkK / kGroupSize, 1);
  static constexpr uint32_t kMfmaPerGroup = kIsChannelScale ? kMfmaPerChunk : MIN(kGroupSize / Op::K, kMfmaPerChunk);
  static constexpr bool kLdsA = TileConfig::kLdsA;
  static constexpr bool kScaleOnB = !kIsChannelScale && TileConfig::kScaleB;
  // Group scales on a temporary FP32 accumulator (exact (code - zp), FP32 scaling).
  static constexpr bool kScaleOnC = !kIsChannelScale && !kScaleOnB;
  static constexpr uint32_t kLdsAUnits = TileConfig::kLdsAUnitsPerThread;
  // LDS A tile: kBlockM rows x 16 units of 16 bytes; unit u of row r is stored at slot
  // u ^ (r % 16) so both the row-wise stores and the column-wise MFMA fragment reads
  // (16 rows, same unit) hit 16 distinct 16-byte bank groups.
  static constexpr uint32_t kLdsABufferUnits = TileConfig::kBlockM * 16;
  // Vector memory loads issued by one load_chunk (vmcnt bookkeeping for the LDS A buffers).
  static constexpr uint32_t kLoadsPerChunk = kTilesN * ((kBits + 3) / 4) +
      (kIsChannelScale ? 0 : kTilesN * kGroupsPerChunk * (1 + kHasZeroPoint)) +
      (kLdsA ? kLdsAUnits : kTilesM * kMfmaPerChunk);

  const uint16_t *a_ptr;
  const uint32_t *b_ptr;
  const uint16_t *bs_ptr;
  const uint8_t *bzp_ptr;
  uint32_t shape_m;
  uint32_t shape_n;
  uint32_t shape_k;
  uint32_t lane;
  uint32_t m_offset;
  uint32_t n_tile_offset;
  uint32_t thread_id;
  uint32_t block_m_offset;
  uint32_t local_m_offset;
  uint32_t warp_id;  // wave-uniform (readfirstlane)
  // One LDS A buffer per stage, each its own __shared__ object: the compiler can then prove that
  // the direct-to-LDS loads of the next chunk do not alias this chunk's ds_reads (otherwise it
  // waits for them with vmcnt(0) before every read).
  mfma_u32x4_t *lds_a[kStages];

  uint32_t regs_qb[kStages][kTilesN][kBits];
  mfma_u32x4_t regs_a[kStages][kLdsA ? 1 : kTilesM][kLdsA ? 1 : kMfmaPerChunk];
  uint16_t regs_bs[kStages][kTilesN][kGroupsPerChunk];
  uint32_t regs_bzp[kStages][kTilesN][kGroupsPerChunk];
  CRegisters regs_c[kTilesM][kTilesN];
  CRegisters regs_group_c[kScaleOnC ? kTilesM : 1][kScaleOnC ? kTilesN : 1];

  CUDA_INLINE bool is_tile_n_valid(uint32_t tile_n) const {
    return (n_tile_offset + tile_n) * 16 < shape_n;
  }

  CUDA_INLINE uint32_t column(uint32_t tile_n) const {
    return (n_tile_offset + tile_n) * 16 + Op::c_col(lane);
  }

  // Out-of-range N tiles / M rows are clamped to valid addresses instead of predicated:
  // their results are never stored, and branch-free loads let the waitcnt pass keep the
  // register prefetch in flight (a guarded load forces s_waitcnt vmcnt(0) at the join).
  CUDA_INLINE uint32_t clamped_n_tile(uint32_t tile_n) const {
    return MIN(n_tile_offset + tile_n, shape_n / 16 - 1);
  }

  CUDA_INLINE uint32_t clamped_column(uint32_t tile_n) const {
    return clamped_n_tile(tile_n) * 16 + Op::c_col(lane);
  }

  // All loads are raw buffer loads (see mfma_buffer_load): the descriptor holds a
  // wave/block-uniform base, the per-lane offset is loop invariant and the per-chunk
  // step goes into the SGPR soffset, so the address math leaves the VALU.
  CUDA_INLINE uint32_t first_n_tile() const {
    return MIN(n_tile_offset, shape_n / 16 - 1);
  }

  CUDA_INLINE void load_chunk(uint32_t stage, uint32_t chunk_k) {
    const uint32_t num_chunks_k = shape_k / kChunkK;
    constexpr uint32_t kChunkBytesB = kWaveSize * kBits * 4;
    const auto rsrc_b = mfma_make_buffer(b_ptr + static_cast<size_t>(first_n_tile()) * num_chunks_k * kWaveSize * kBits);
    PRAGMA_UNROLL
    for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
      const uint32_t voffset = ((clamped_n_tile(tile_n) - first_n_tile()) * num_chunks_k * kWaveSize + lane) * kBits * 4;
      static_for<0, (kBits + 3) / 4>([&](auto part) {
        constexpr uint32_t i = decltype(part)::value * 4;
        mfma_buffer_load<MIN(kBits - i, 4)>(rsrc_b, voffset + i * 4, chunk_k * kChunkBytesB, &regs_qb[stage][tile_n][i]);
      });

      if constexpr (!kIsChannelScale) {
        const auto rsrc_bs = mfma_make_buffer(bs_ptr);
        PRAGMA_UNROLL
        for (uint32_t group = 0; group < kGroupsPerChunk; group++) {
          const uint32_t group_index = (chunk_k * kChunkK + group * MIN(kGroupSize, kChunkK)) / kGroupSize;
          const uint32_t column_offset = clamped_column(tile_n);
          regs_bs[stage][tile_n][group] =
              __builtin_amdgcn_raw_buffer_load_b16(rsrc_bs, column_offset * 2, group_index * shape_n * 2, 0);
          if constexpr (kHasZeroPoint) {
            const auto rsrc_bzp = mfma_make_buffer(bzp_ptr);
            regs_bzp[stage][tile_n][group] =
                __builtin_amdgcn_raw_buffer_load_b8(rsrc_bzp, column_offset, group_index * shape_n, 0);
          }
        }
      }
    }

    // A rows past M re-read row M - 1: A row r only feeds C row r, which is not stored.
    const auto rsrc_a = mfma_make_buffer(a_ptr + static_cast<size_t>(block_m_offset) * shape_k);
    const uint32_t soffset_a = chunk_k * kChunkK * 2;
    if constexpr (kLdsA) {
      // Direct-to-LDS loads: lane l of a wave writes LDS slot (wave base + l), so the swizzle
      // is applied on the global side: slot s of row r holds 16-byte unit s ^ (r % 16).
      constexpr uint32_t kNumWaves = TileConfig::kNumThreads / kWaveSize;
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLdsAUnits; i++) {
        const uint32_t wave_slot = (i * kNumWaves + warp_id) * kWaveSize;
        const uint32_t slot = wave_slot + lane;
        const uint32_t row = slot / 16;
        const uint32_t unit = slot % 16 ^ row % 16;
        const uint32_t global_row = MIN(block_m_offset + row, shape_m - 1) - block_m_offset;
        const uint32_t voffset = (global_row * shape_k + unit * 8) * 2;
        mfma_u32x4_t *dst = lds_a[stage] + wave_slot;
        mfma_buffer_load_lds_b128(rsrc_a, dst, voffset, soffset_a);
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
        const uint32_t row = MIN(m_offset + tile_m * 16 + Op::a_row(lane), shape_m - 1) - block_m_offset;
        const uint32_t voffset = (row * shape_k + Op::ab_k(lane, 0)) * 2;
        PRAGMA_UNROLL
        for (uint32_t mfma_index = 0; mfma_index < kMfmaPerChunk; mfma_index++) {
          regs_a[stage][tile_m][mfma_index] =
              __builtin_amdgcn_raw_buffer_load_b128(rsrc_a, voffset + mfma_index * Op::K * 2, soffset_a, 0);
        }
      }
    }
  }

  CUDA_INLINE void apply_group_scale(uint32_t stage, uint32_t group) {
    PRAGMA_UNROLL
    for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
      const float scale = mfma_half_to_float<ElementA>(regs_bs[stage][tile_n][group]);
      PRAGMA_UNROLL
      for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
        PRAGMA_UNROLL
        for (uint32_t reg = 0; reg < Op::kRegsC; reg++) {
          regs_c[tile_m][tile_n][reg] += regs_group_c[tile_m][tile_n][reg] * scale;
          regs_group_c[tile_m][tile_n][reg] = 0.0f;
        }
      }
    }
  }

  CUDA_INLINE void compute_chunk(uint32_t stage, uint32_t chunk_k, uint32_t *channel_zero_points) {
    const mfma_u32x4_t *lds_buffer = lds_a[stage];
    PRAGMA_UNROLL
    for (uint32_t mfma_index = 0; mfma_index < kMfmaPerChunk; mfma_index++) {
      mfma_u32x4_t lds_regs_a[kLdsA ? kTilesM : 1];
      if constexpr (kLdsA) {
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          const uint32_t row = local_m_offset + tile_m * 16 + Op::a_row(lane);
          const uint32_t unit = mfma_index * 4 + lane / 16;
          lds_regs_a[tile_m] = lds_buffer[row * 16 + (unit ^ row % 16)];
        }
      }
      const uint32_t group = mfma_index / kMfmaPerGroup % kGroupsPerChunk;
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        uint32_t zero_point = 0;
        if constexpr (kHasZeroPoint && kIsChannelScale) {
          zero_point = channel_zero_points[tile_n];
        } else if constexpr (kHasZeroPoint) {
          zero_point = Dequant::prepare_zero_point(regs_bzp[stage][tile_n][group]);
        }
        uint32_t regs_b[Op::kRegsAB];
        if constexpr (kScaleOnB) {
          Dequant::run_scaled(regs_qb[stage][tile_n], regs_b, mfma_index, zero_point, regs_bs[stage][tile_n][group]);
        } else {
          Dequant::run(regs_qb[stage][tile_n], regs_b, mfma_index, zero_point);
        }
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          const uint32_t *regs_a_ptr;
          if constexpr (kLdsA) {
            regs_a_ptr = reinterpret_cast<const uint32_t *>(&lds_regs_a[tile_m]);
          } else {
            regs_a_ptr = reinterpret_cast<const uint32_t *>(&regs_a[stage][tile_m][mfma_index]);
          }
          if constexpr (!kScaleOnC) {
            Op::fma(regs_a_ptr, regs_b, regs_c[tile_m][tile_n]);
          } else {
            Op::fma(regs_a_ptr, regs_b, regs_group_c[tile_m][tile_n]);
          }
        }
      }
      if constexpr (kScaleOnC && kGroupSize < kChunkK) {
        if ((mfma_index + 1) % kMfmaPerGroup == 0) apply_group_scale(stage, group);
      }
    }
    if constexpr (kScaleOnC && kGroupSize >= kChunkK) {
      if ((chunk_k + 1) * kChunkK % kGroupSize == 0) apply_group_scale(stage, 0);
    }
  }

  CUDA_INLINE void mainloop(uint32_t chunk_begin, uint32_t chunk_end) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        regs_c[tile_m][tile_n] = CRegisters{};
        if constexpr (kScaleOnC) regs_group_c[tile_m][tile_n] = CRegisters{};
      }
    }

    uint32_t channel_zero_points[kTilesN];
    if constexpr (kHasZeroPoint && kIsChannelScale) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        channel_zero_points[tile_n] = is_tile_n_valid(tile_n) ? Dequant::prepare_zero_point(bzp_ptr[column(tile_n)]) : 0;
      }
    }

    const uint32_t num_iters = chunk_end - chunk_begin;
    if (num_iters == 0) return;
    // Prefetches past the end of this K range re-load the last chunk (an L2 hit) so that
    // every load is unconditional; see clamped_n_tile.
    const uint32_t last_chunk = chunk_end - 1;
    static_for<0, kStages - 1>([&](auto stage) {
      load_chunk(decltype(stage)::value, MIN(chunk_begin + decltype(stage)::value, last_chunk));
    });

    uint32_t base = 0;
    for (; base + kStages <= num_iters; base += kStages) {
      static_for<0, kStages>([&](auto stage_constant) {
        constexpr uint32_t stage = decltype(stage_constant)::value;
        constexpr uint32_t load_stage = (stage + kStages - 1) % kStages;
        if constexpr (kLdsA) {
          // This chunk's LDS A has landed for every wave (only the later kStages - 2 chunks may
          // still be in flight), and every wave is done with the buffer the next load overwrites.
          mfma_wait_vmcnt<(kStages - 2) * kLoadsPerChunk>();
          __builtin_amdgcn_s_barrier();
        }
        load_chunk(load_stage, MIN(chunk_begin + base + stage + kStages - 1, last_chunk));
        // Keep the prefetch issued ahead of this chunk's math; otherwise the scheduler
        // sinks the loads to the end of the unrolled body and the pipeline gets shallower.
        __builtin_amdgcn_sched_barrier(0);
        compute_chunk(stage, chunk_begin + base + stage, channel_zero_points);
      });
    }
    // Tail (< kStages chunks): all remaining data is already in flight.
    static_for<0, kStages - 1>([&](auto stage_constant) {
      constexpr uint32_t stage = decltype(stage_constant)::value;
      if (base + stage < num_iters) {
        if constexpr (kLdsA) {
          mfma_wait_vmcnt<(kStages - 2 - stage) * kLoadsPerChunk>();
          __builtin_amdgcn_s_barrier();
        }
        compute_chunk(stage, chunk_begin + base + stage, channel_zero_points);
      }
    });

    if constexpr (kScaleOnC && kGroupSize > kChunkK) {
      // A K split that ends inside a group still owes that group's scale.
      if (num_iters > 0 && chunk_end * kChunkK % kGroupSize != 0) {
        PRAGMA_UNROLL
        for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
          const uint32_t group_index = (chunk_end - 1) * kChunkK / kGroupSize;
          regs_bs[0][tile_n][0] = is_tile_n_valid(tile_n) ? bs_ptr[static_cast<size_t>(group_index) * shape_n + column(tile_n)] : 0;
        }
        apply_group_scale(0, 0);
      }
    }

    if constexpr (kIsChannelScale) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        if (!is_tile_n_valid(tile_n)) continue;
        const float scale = mfma_half_to_float<ElementA>(bs_ptr[column(tile_n)]);
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          PRAGMA_UNROLL
          for (uint32_t reg = 0; reg < Op::kRegsC; reg++) regs_c[tile_m][tile_n][reg] *= scale;
        }
      }
    }
  }

  template <class Func>
  CUDA_INLINE void for_each_output(Func &&func) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        if (!is_tile_n_valid(tile_n)) continue;
        PRAGMA_UNROLL
        for (uint32_t reg = 0; reg < Op::kRegsC; reg++) {
          const uint32_t row = m_offset + tile_m * 16 + Op::c_row(lane, reg);
          if (row < shape_m) func(row, column(tile_n), regs_c[tile_m][tile_n][reg]);
        }
      }
    }
  }
};


// Min 2 blocks per CU caps VGPR + AGPR at 256 per lane (2 waves/SIMD for 4-wave blocks). Without it the
// bf16 8-row tile allocated 170 VGPR + 96 AGPR and fell to occupancy 1; fp16 is unaffected.
template <class ElementA, class ElementQB, class TileConfig, uint32_t kGroupSize, bool kHasZeroPoint>
__global__ __launch_bounds__(TileConfig::kNumThreads, 2) void humming_mfma(
    const uint16_t *__restrict__ A,
    const uint32_t *__restrict__ B,
    const uint16_t *__restrict__ BS,
    const uint8_t *__restrict__ BZP,
    uint16_t *__restrict__ C,
    float *__restrict__ workspace,
    uint32_t *__restrict__ tile_counters,
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k) {
  using Gemm = MfmaWeightOnlyGemm<ElementA, ElementQB, TileConfig, kGroupSize, kHasZeroPoint>;
  // readfirstlane: tells the compiler the wave index is uniform, so buffer descriptors
  // derived from it stay in SGPRs (otherwise it emits a waterfall loop per load).
  const uint32_t warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize);
  const uint32_t warp_m = warp_id / TileConfig::kWarpsN;
  const uint32_t warp_n = warp_id % TileConfig::kWarpsN;

  constexpr uint32_t kLdsAUnits = TileConfig::kLdsA ? TileConfig::kBlockM * 16 : 1;
  __shared__ mfma_u32x4_t lds_a0[kLdsAUnits];
  __shared__ mfma_u32x4_t lds_a1[TileConfig::kStages > 1 ? kLdsAUnits : 1];
  __shared__ mfma_u32x4_t lds_a2[TileConfig::kStages > 2 ? kLdsAUnits : 1];
  __shared__ mfma_u32x4_t lds_a3[TileConfig::kStages > 3 ? kLdsAUnits : 1];
  static_assert(TileConfig::kStages <= 4);
  Gemm gemm;
  mfma_u32x4_t *lds_buffers[4] = {lds_a0, lds_a1, lds_a2, lds_a3};
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < TileConfig::kStages; i++) gemm.lds_a[i] = lds_buffers[i];
  gemm.thread_id = threadIdx.x;
  gemm.warp_id = warp_id;
  gemm.block_m_offset = blockIdx.y * TileConfig::kBlockM;
  gemm.local_m_offset = warp_m * TileConfig::kWarpTilesM * 16;
  gemm.a_ptr = A;
  gemm.b_ptr = B;
  gemm.bs_ptr = BS;
  gemm.bzp_ptr = BZP;
  gemm.shape_m = shape_m;
  gemm.shape_n = shape_n;
  gemm.shape_k = shape_k;
  gemm.lane = threadIdx.x % kWaveSize;
  gemm.m_offset = blockIdx.y * TileConfig::kBlockM + warp_m * TileConfig::kWarpTilesM * 16;
  gemm.n_tile_offset = (blockIdx.x * TileConfig::kBlockN + warp_n * TileConfig::kWarpTilesN * 16) / 16;

  const uint32_t num_chunks_k = shape_k / Gemm::kChunkK;
  const uint32_t num_splits = gridDim.z;
  const uint32_t chunk_begin = blockIdx.z * num_chunks_k / num_splits;
  const uint32_t chunk_end = (blockIdx.z + 1) * num_chunks_k / num_splits;
  gemm.mainloop(chunk_begin, chunk_end);

  if (num_splits == 1) {
    gemm.for_each_output([&](uint32_t row, uint32_t col, float value) {
      C[static_cast<size_t>(row) * shape_n + col] = mfma_float_to_half<ElementA>(value);
    });
    return;
  }

  // Split-K: publish FP32 partials, the last block of this output tile reduces them.
  //
  // An agent-scope release/acquire fence on gfx950 is a full L2 writeback + invalidate
  // (buffer_wbl2 sc1 / buffer_inv sc1, L2 is per XCD), which cost ~4 us per split. Instead
  // every partial is stored / loaded with the sc1 cache policy (what agent-scope atomics
  // use), which is coherent across XCDs on its own; we only have to make sure the stores
  // have completed (s_waitcnt vmcnt(0), gfx9 counts stores in vmcnt) before the counter
  // increment.
  //
  // Partials keep the accumulator fragment layout, [split][tile][tile_m][tile_n][thread]
  // of 16 bytes: every store/load is a fully coalesced dwordx4, and the last block adds
  // the other splits straight into its registers.
  constexpr uint32_t kTileFloats = TileConfig::kBlockM * TileConfig::kBlockN;
  constexpr uint32_t kSc1 = 16;  // gfx940+ CPol::SC1
  const uint32_t tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  const uint32_t num_tiles = gridDim.x * gridDim.y;
  auto fragment_offset = [&](uint32_t tile_m, uint32_t tile_n) {
    return ((tile_m * Gemm::kTilesN + tile_n) * TileConfig::kNumThreads + threadIdx.x) * 16;
  };
  auto split_buffer = [&](uint32_t split) {
    return mfma_make_buffer(workspace + (static_cast<size_t>(split) * num_tiles + tile_id) * kTileFloats, kTileFloats * 4);
  };
  {
    const auto rsrc = split_buffer(blockIdx.z);
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < Gemm::kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < Gemm::kTilesN; tile_n++) {
        __builtin_amdgcn_raw_buffer_store_b128(gemm.regs_c[tile_m][tile_n], rsrc, fragment_offset(tile_m, tile_n), 0, kSc1);
      }
    }
  }
  __builtin_amdgcn_s_waitcnt(0x0F70);  // vmcnt(0) expcnt(7) lgkmcnt(15)
  __syncthreads();

  __shared__ uint32_t is_last_block;
  if (threadIdx.x == 0) {
    uint32_t arrived = __hip_atomic_fetch_add(&tile_counters[tile_id], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    is_last_block = arrived == num_splits - 1;
  }
  __syncthreads();
  if (!is_last_block) return;

  // Re-read this block's own partial too: keeping regs_c live next to the incoming
  // partials raises the kernel's VGPR peak (and halves occupancy for large tiles).
  // Steps of kRowsPerStep fragment rows bound the epilogue registers; every load of a row,
  // across all splits, is in flight at once.
  constexpr uint32_t kRowsPerStep = MIN(Gemm::kTilesM, 4u);
  PRAGMA_UNROLL
  for (uint32_t tile_m_base = 0; tile_m_base < Gemm::kTilesM; tile_m_base += kRowsPerStep) {
    typename Gemm::CRegisters sums[kRowsPerStep][Gemm::kTilesN] = {};
    for (uint32_t split = 0; split < num_splits; split++) {
      const auto rsrc = split_buffer(split);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kRowsPerStep; i++) {
        PRAGMA_UNROLL
        for (uint32_t tile_n = 0; tile_n < Gemm::kTilesN; tile_n++) {
          sums[i][tile_n] += __builtin_amdgcn_raw_buffer_load_b128(rsrc, fragment_offset(tile_m_base + i, tile_n), 0, kSc1);
        }
      }
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kRowsPerStep; i++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < Gemm::kTilesN; tile_n++) gemm.regs_c[tile_m_base + i][tile_n] = sums[i][tile_n];
    }
  }
  gemm.for_each_output([&](uint32_t row, uint32_t col, float value) {
    C[static_cast<size_t>(row) * shape_n + col] = mfma_float_to_half<ElementA>(value);
  });
  if (threadIdx.x == 0) tile_counters[tile_id] = 0;
}

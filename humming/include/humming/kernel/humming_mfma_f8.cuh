#pragma once

// Low-bit activation GEMM for AMD CDNA4 (gfx950):
//
//   C[M, N] = epilogue(A[M, K] @ dequant(B)[N, K]^T)
//
// A: FP8 (e4m3 / e5m2) or INT8 bytes [M, K], or packed FP4 e2m1 [M, K / 2] (MXFP4).
// B: per-lane chunks of 16 channels x 128 K (humming/rocm/layout_f8.py), either fed to the MFMA
//    as is (FP8 / FP6 / FP4 / INT8 "native" weights) or dequantized to FP8 e4m3 / INT8 with a
//    byte-wise table lookup (INT1-5 and small FP codes) or byte arithmetic (INT codes -> INT8).
// MFMA: v_mfma_scale_f32_16x16x128_f8f6f4 (FP8 / FP4 activations, E8M0 MX block scales in
//    hardware), v_mfma_i32_16x16x64_i8 (INT8), or the K = 32 FP8 / INT8 MFMAs when FP32 scales
//    change every 32 K (fragment layouts in mma/mfma_f8f6f4.cuh).
// Scales: FP32 group scales (weight and / or activation) are applied by flushing the MFMA
//    accumulator into an FP32 accumulator at every scale-group boundary; E8M0 group scales go
//    to the MFMA scale operands when the MFMA is the K = 128 scaled one; per-token / per-tensor
//    activation scales and per-channel / per-tensor weight scales (folded into one FP32 [N]
//    vector) are applied in the epilogue.
//
// The block / wave / split-K structure follows humming_mfma.cuh (kStages-deep register
// prefetch, optional LDS staging of A, deterministic split-K reduction by the last block).

#include <humming/kernel/humming_mfma.cuh>
#include <humming/mma/mfma_f8f6f4.cuh>
#include <humming/utils/all.cuh>

// B handling.
constexpr uint32_t kMfmaF8BNative = 0;   // stored bytes are the MFMA operand
constexpr uint32_t kMfmaF8BLut = 1;      // codes -> table lookup -> FP8 / INT8 bytes
constexpr uint32_t kMfmaF8BIntArith = 2; // unsigned codes -> signed INT8 (code - zero point)

// Weight / activation scale modes.
constexpr uint32_t kMfmaF8ScaleNone = 0;      // weight: folded into the FP32 [N] epilogue vector
                                              // activation: FP32 per row (token / tensor) epilogue
constexpr uint32_t kMfmaF8ScaleFlush = 1;     // FP32 group scale, applied at accumulator flushes
constexpr uint32_t kMfmaF8ScaleHw = 2;        // E8M0 group scale on the MFMA scale operand
constexpr uint32_t kMfmaF8ScaleE8m0Flush = 3; // E8M0 group scale, converted, applied at flushes

template <
    uint32_t kWarpsM_, uint32_t kWarpsN_, uint32_t kWarpTilesM_, uint32_t kWarpTilesN_, uint32_t kStages_,
    bool kLdsA_ = false>
struct MfmaF8TileConfig {
  static constexpr uint32_t kWarpsM = kWarpsM_;
  static constexpr uint32_t kWarpsN = kWarpsN_;
  static constexpr uint32_t kWarpTilesM = kWarpTilesM_;
  static constexpr uint32_t kWarpTilesN = kWarpTilesN_;
  static constexpr uint32_t kStages = kStages_;
  static constexpr uint32_t kNumThreads = kWarpsM * kWarpsN * kWaveSize;
  static constexpr uint32_t kBlockM = kWarpsM * kWarpTilesM * 16;
  static constexpr uint32_t kBlockN = kWarpsN * kWarpTilesN * 16;
  static constexpr bool kLdsA = kLdsA_;
  static_assert(kStages >= 2, "register prefetch needs at least two stages");
};


// Bit planes of the dequantized B codes: a code of kBits bits is split into planes of 4 / 2 / 1
// bits (low bits first). Plane p of width w occupies w consecutive words of a lane's chunk; in
// those words, value j = 4 r + b (register r, byte b) sits at bit 8 b + w * (r % (8 / w)) of word
// r / (8 / w), so one shift + one mask extracts 4 byte-aligned values (layout_f8.py mirrors this).
template <uint32_t kBits>
struct MfmaF8Planes {
  static constexpr uint32_t kHas8 = kBits == 8;
  static constexpr uint32_t kHas4 = kBits < 8 && kBits >= 4;
  static constexpr uint32_t kHas2 = kBits < 8 && (kBits % 4) >= 2;
  static constexpr uint32_t kHas1 = kBits < 8 && (kBits % 2) == 1;

  template <uint32_t kWidth>
  CUDA_INLINE static uint32_t plane(const uint32_t *qb, uint32_t word_base, uint32_t reg) {
    constexpr uint32_t kPerWord = 8 / kWidth;
    constexpr uint32_t kMask = ((1u << kWidth) - 1) * 0x01010101u;
    return (qb[word_base + reg / kPerWord] >> (kWidth * (reg % kPerWord))) & kMask;
  }

  // Codes of values 4 reg .. 4 reg + 3, one per byte.
  CUDA_INLINE static uint32_t codes(const uint32_t *qb, uint32_t reg) {
    if constexpr (kHas8) {
      return qb[reg];
    } else {
      uint32_t code = 0;
      uint32_t word_base = 0;
      uint32_t bit_offset = 0;
      if constexpr (kHas4) {
        code = plane<4>(qb, word_base, reg);
        word_base += 4, bit_offset += 4;
      }
      if constexpr (kHas2) {
        code |= plane<2>(qb, word_base, reg) << bit_offset;
        word_base += 2, bit_offset += 2;
      }
      if constexpr (kHas1) {
        code |= plane<1>(qb, word_base, reg) << bit_offset;
      }
      return code;
    }
  }
};


template <class Traits, class TileConfig>
struct MfmaF8Gemm {
  using ElementC = typename Traits::ElementC;

  static constexpr uint32_t kAFmt = Traits::kAFmt;
  static constexpr uint32_t kBFmt = Traits::kBFmt;
  static constexpr uint32_t kBMode = Traits::kBMode;
  static constexpr uint32_t kBits = Traits::kBits;
  static constexpr bool kHasZeroPoint = Traits::kHasZeroPoint;
  static constexpr uint32_t kMfmaK = Traits::kMfmaK;
  static constexpr uint32_t kScaleK = Traits::kScaleK;
  static constexpr uint32_t kWsMode = Traits::kWsMode;
  static constexpr uint32_t kWsGroup = Traits::kWsGroup;
  static constexpr uint32_t kZpGroup = Traits::kZpGroup;
  static constexpr uint32_t kAsMode = Traits::kAsMode;
  static constexpr uint32_t kAsGroup = Traits::kAsGroup;

  static constexpr bool kIsInt8 = kAFmt == kMfmaFmtInt8;
  static constexpr bool kIsFp4A = kAFmt == kMfmaFmtE2M1;
  static_assert(kMfmaK == 128 || kMfmaK == 64 || kMfmaK == 32);
  static_assert(kIsInt8 ? kMfmaK != 128 : kMfmaK != 64);
  static_assert(kIsInt8 == (kBFmt == kMfmaFmtInt8));
  static_assert(!kIsFp4A || kMfmaK == 128);
  static constexpr bool kIsL32 = kMfmaK == 32;
  static constexpr bool kIsNativeTrueB = kBMode == kMfmaF8BNative && (kBFmt == kMfmaFmtE2M1 || kBFmt == kMfmaFmtE2M3 || kBFmt == kMfmaFmtE3M2);
  static_assert(!kIsNativeTrueB || kMfmaK == 128);
  static_assert(kWsMode != kMfmaF8ScaleHw || kMfmaK == 128);
  static_assert(kAsMode != kMfmaF8ScaleHw || kMfmaK == 128);

  static constexpr uint32_t kChunkK = 128;
  static constexpr uint32_t kUnits = kChunkK / kMfmaK;  // MFMAs per (tile_m, tile_n) per chunk
  static constexpr uint32_t kARegs = kIsFp4A ? 4 : 8;   // A registers per lane per chunk
  static constexpr uint32_t kBRegs = mfma_fmt_regs(kBFmt);
  static constexpr uint32_t kARegsPerUnit = kARegs / kUnits;
  static constexpr uint32_t kBRegsPerUnit = kBRegs / kUnits;
  static constexpr uint32_t kAChunkBytes = kIsFp4A ? 64 : 128;
  static constexpr uint32_t kBWords = kBits;  // 32 values x kBits bits per lane per chunk

  static constexpr bool kHasFlushScale = kScaleK > 0;
  // Accumulator flushes inside one chunk (kScaleK < 128) or at most one per chunk.
  static constexpr uint32_t kFlushK = kHasFlushScale ? MIN(kScaleK, kChunkK) : kChunkK;
  static constexpr uint32_t kFlushesPerChunk = kChunkK / kFlushK;
  static constexpr uint32_t kUnitsPerFlush = kFlushK / kMfmaK;
  static_assert(!kHasFlushScale || (kScaleK % kMfmaK == 0 && (kScaleK < kChunkK ? kChunkK % kScaleK == 0 : kScaleK % kChunkK == 0)));
  static constexpr bool kWsFlush = kWsMode == kMfmaF8ScaleFlush || kWsMode == kMfmaF8ScaleE8m0Flush;
  static constexpr bool kAsFlush = kAsMode == kMfmaF8ScaleFlush || kAsMode == kMfmaF8ScaleE8m0Flush;

  // Zero points: one per lane per "K block" of its 32 B values (L128: two halves of 16 values,
  // L32: four blocks of 8 values), deduplicated when the zero-point group spans the chunk.
  static constexpr uint32_t kZpBlocks = !kHasZeroPoint ? 1 : (kZpGroup == 0 || kZpGroup >= kChunkK) ? 1 : (kIsL32 ? 4 : 2);
  static constexpr uint32_t kZpRegsPerBlock = kBRegs / kZpBlocks;

  // LDS A tile: kBlockM rows x kRowUnits 16-byte units per chunk, unit u of row r stored at
  // u ^ (r % kRowUnits).
  static constexpr bool kLdsA = TileConfig::kLdsA;
  static constexpr uint32_t kRowUnits = kAChunkBytes / 16;
  static constexpr uint32_t kLdsBufferUnits = TileConfig::kBlockM * kRowUnits;
  static constexpr uint32_t kLdsAUnits = kLdsA ? kLdsBufferUnits / TileConfig::kNumThreads : 1;
  static_assert(!kLdsA || kLdsBufferUnits % TileConfig::kNumThreads == 0);
  // FP32 activation group scales staged through LDS next to A: one buffer load per block row and
  // chunk (instead of 4 per lane and 16-row tile), read back as one 16-byte LDS load per tile.
  static constexpr bool kLdsAs = kLdsA && kAsMode == kMfmaF8ScaleFlush;
  static constexpr uint32_t kLdsAsWords = TileConfig::kBlockM * kFlushesPerChunk;
  static constexpr uint32_t kAsRowsPerThread = CEIL_DIV(TileConfig::kBlockM, TileConfig::kNumThreads);

  static constexpr uint32_t kTilesM = TileConfig::kWarpTilesM;
  static constexpr uint32_t kTilesN = TileConfig::kWarpTilesN;
  static constexpr uint32_t kStages = TileConfig::kStages;

  using Acc = std::conditional_t<kIsInt8, mfma_i32x4_t, mfma_f32x4_t>;
  using Planes = MfmaF8Planes<kBits>;

  const uint8_t *a_ptr;
  const uint32_t *b_ptr;
  const void *ws_ptr;
  const uint8_t *bzp_ptr;
  const void *as_ptr;
  uint32_t as_stride;
  uint32_t shape_m;
  uint32_t shape_n;
  uint32_t shape_k;
  uint32_t lane;
  uint32_t m_offset;
  uint32_t n_tile_offset;
  uint32_t thread_id;
  uint32_t block_m_offset;
  uint32_t local_m_offset;
  mfma_u32x4_t *lds_a;
  float *lds_as;

  uint32_t regs_qb[kStages][kTilesN][kBWords];
  uint32_t regs_a[kStages][kLdsA ? 1 : kTilesM][kLdsA ? 1 : kARegs];
  mfma_u32x4_t regs_a_share[kStages][kLdsAUnits];
  uint32_t regs_ws[kStages][kTilesN][kWsFlush ? kFlushesPerChunk : 1];
  uint32_t regs_wsb[kStages][kTilesN];
  uint32_t regs_as[kStages][kAsFlush && !kLdsAs ? kTilesM : 1][kAsFlush && !kLdsAs ? kFlushesPerChunk : 1][kAsFlush && !kLdsAs ? 4 : 1];
  uint32_t regs_as_row[kStages][kLdsAs ? kAsRowsPerThread : 1][kLdsAs ? kFlushesPerChunk : 1];
  uint32_t regs_asb[kStages][kTilesM];
  uint32_t regs_bzp[kStages][kTilesN][kZpBlocks];
  Acc regs_part[kIsInt8 && !kHasFlushScale ? kTilesM : 1][kIsInt8 && !kHasFlushScale ? kTilesN : 1];
  mfma_f32x4_t regs_c[kTilesM][kTilesN];

  CUDA_INLINE bool is_tile_n_valid(uint32_t tile_n) const {
    return (n_tile_offset + tile_n) * 16 < shape_n;
  }
  CUDA_INLINE uint32_t column(uint32_t tile_n) const {
    return (n_tile_offset + tile_n) * 16 + lane % 16;
  }
  CUDA_INLINE uint32_t clamped_n_tile(uint32_t tile_n) const {
    return MIN(n_tile_offset + tile_n, shape_n / 16 - 1);
  }
  CUDA_INLINE uint32_t clamped_column(uint32_t tile_n) const {
    return clamped_n_tile(tile_n) * 16 + lane % 16;
  }
  CUDA_INLINE uint32_t first_n_tile() const {
    return MIN(n_tile_offset, shape_n / 16 - 1);
  }

  // K offset (within a chunk) of the first value of zero-point block `block` of this lane.
  CUDA_INLINE uint32_t zp_block_k(uint32_t block) const {
    if constexpr (kZpBlocks == 1) {
      return 0;
    } else if constexpr (kIsL32) {
      return block * 32 + lane / 16 * 8;
    } else {
      return block * 64 + lane / 16 * 16;
    }
  }

  CUDA_INLINE void load_chunk(uint32_t stage, uint32_t chunk_k) {
    const uint32_t num_chunks_k = shape_k / kChunkK;
    constexpr uint32_t kChunkBytesB = kWaveSize * kBWords * 4;
    const auto rsrc_b = mfma_make_buffer(b_ptr + static_cast<size_t>(first_n_tile()) * num_chunks_k * kWaveSize * kBWords);
    const auto rsrc_ws = mfma_make_buffer(ws_ptr);
    const auto rsrc_bzp = mfma_make_buffer(bzp_ptr);
    PRAGMA_UNROLL
    for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
      const uint32_t voffset = ((clamped_n_tile(tile_n) - first_n_tile()) * num_chunks_k * kWaveSize + lane) * kBWords * 4;
      static_for<0, (kBWords + 3) / 4>([&](auto part) {
        constexpr uint32_t i = decltype(part)::value * 4;
        mfma_buffer_load<MIN(kBWords - i, 4)>(rsrc_b, voffset + i * 4, chunk_k * kChunkBytesB, &regs_qb[stage][tile_n][i]);
      });
      const uint32_t col = clamped_column(tile_n);
      if constexpr (kWsMode == kMfmaF8ScaleFlush) {
        PRAGMA_UNROLL
        for (uint32_t f = 0; f < kFlushesPerChunk; f++) {
          const uint32_t group = (chunk_k * kChunkK + f * kFlushK) / kWsGroup;
          regs_ws[stage][tile_n][f] = __builtin_amdgcn_raw_buffer_load_b32(rsrc_ws, col * 4, group * shape_n * 4, 0);
        }
      } else if constexpr (kWsMode == kMfmaF8ScaleE8m0Flush) {
        PRAGMA_UNROLL
        for (uint32_t f = 0; f < kFlushesPerChunk; f++) {
          const uint32_t group = (chunk_k * kChunkK + f * kFlushK) / kWsGroup;
          regs_ws[stage][tile_n][f] = __builtin_amdgcn_raw_buffer_load_b8(rsrc_ws, col, group * shape_n, 0);
        }
      } else if constexpr (kWsMode == kMfmaF8ScaleHw) {
        // Lane group lg scales K block [32 lg, 32 lg + 32) of the chunk.
        const uint32_t group = (chunk_k * kChunkK + lane / 16 * 32) / kWsGroup;
        regs_wsb[stage][tile_n] = __builtin_amdgcn_raw_buffer_load_b8(rsrc_ws, group * shape_n + col, 0, 0);
      }
      if constexpr (kHasZeroPoint) {
        PRAGMA_UNROLL
        for (uint32_t block = 0; block < kZpBlocks; block++) {
          const uint32_t group = kZpGroup == 0 ? 0 : (chunk_k * kChunkK + zp_block_k(block)) / kZpGroup;
          regs_bzp[stage][tile_n][block] = __builtin_amdgcn_raw_buffer_load_b8(rsrc_bzp, group * shape_n + col, 0, 0);
        }
      }
    }

    // Activation scales.
    if constexpr (kLdsAs) {
      const auto rsrc_as = mfma_make_buffer(as_ptr);
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kAsRowsPerThread; i++) {
        const uint32_t row = MIN(block_m_offset + MIN(thread_id + i * TileConfig::kNumThreads, TileConfig::kBlockM - 1), shape_m - 1);
        PRAGMA_UNROLL
        for (uint32_t f = 0; f < kFlushesPerChunk; f++) {
          const uint32_t group = (chunk_k * kChunkK + f * kFlushK) / kAsGroup;
          regs_as_row[stage][i][f] = __builtin_amdgcn_raw_buffer_load_b32(rsrc_as, (row * as_stride + group) * 4, 0, 0);
        }
      }
    } else if constexpr (kAsFlush || kAsMode == kMfmaF8ScaleHw) {
      const auto rsrc_as = mfma_make_buffer(as_ptr);
      PRAGMA_UNROLL
      for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
        if constexpr (kAsMode == kMfmaF8ScaleHw) {
          const uint32_t row = MIN(m_offset + tile_m * 16 + lane % 16, shape_m - 1);
          const uint32_t group = (chunk_k * kChunkK + lane / 16 * 32) / kAsGroup;
          regs_asb[stage][tile_m] = __builtin_amdgcn_raw_buffer_load_b8(rsrc_as, row * as_stride + group, 0, 0);
        } else {
          PRAGMA_UNROLL
          for (uint32_t f = 0; f < kFlushesPerChunk; f++) {
            const uint32_t group = (chunk_k * kChunkK + f * kFlushK) / kAsGroup;
            PRAGMA_UNROLL
            for (uint32_t r = 0; r < 4; r++) {
              const uint32_t row = MIN(m_offset + tile_m * 16 + lane / 16 * 4 + r, shape_m - 1);
              if constexpr (kAsMode == kMfmaF8ScaleFlush) {
                regs_as[stage][tile_m][f][r] = __builtin_amdgcn_raw_buffer_load_b32(rsrc_as, (row * as_stride + group) * 4, 0, 0);
              } else {
                regs_as[stage][tile_m][f][r] = __builtin_amdgcn_raw_buffer_load_b8(rsrc_as, row * as_stride + group, 0, 0);
              }
            }
          }
        }
      }
    }

    // A rows past M re-read row M - 1: A row r only feeds C row r, which is not stored.
    const uint32_t row_bytes = shape_k * kAChunkBytes / kChunkK;
    const auto rsrc_a = mfma_make_buffer(a_ptr + static_cast<size_t>(block_m_offset) * row_bytes);
    const uint32_t soffset_a = chunk_k * kAChunkBytes;
    if constexpr (kLdsA) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLdsAUnits; i++) {
        const uint32_t unit_index = thread_id + i * TileConfig::kNumThreads;
        const uint32_t row = MIN(block_m_offset + unit_index / kRowUnits, shape_m - 1) - block_m_offset;
        const uint32_t voffset = row * row_bytes + unit_index % kRowUnits * 16;
        regs_a_share[stage][i] = __builtin_amdgcn_raw_buffer_load_b128(rsrc_a, voffset, soffset_a, 0);
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
        const uint32_t row = MIN(m_offset + tile_m * 16 + lane % 16, shape_m - 1) - block_m_offset;
        const uint32_t lg = lane / 16;
        uint32_t *dst = regs_a[stage][tile_m];
        if constexpr (kIsFp4A) {
          mfma_buffer_load<4>(rsrc_a, row * row_bytes + lg * 16, soffset_a, dst);
        } else if constexpr (kIsL32) {
          PRAGMA_UNROLL
          for (uint32_t t = 0; t < 4; t++) mfma_buffer_load<2>(rsrc_a, row * row_bytes + t * 32 + lg * 8, soffset_a, dst + 2 * t);
        } else {
          mfma_buffer_load<4>(rsrc_a, row * row_bytes + lg * 16, soffset_a, dst);
          mfma_buffer_load<4>(rsrc_a, row * row_bytes + 64 + lg * 16, soffset_a, dst + 4);
        }
      }
    }
  }

  // B operand register `reg` (values 4 reg .. 4 reg + 3 of this lane's chunk).
  CUDA_INLINE uint32_t dequant_reg(const uint32_t *qb, uint32_t reg, const uint32_t *lut, const uint32_t *zp_prepared) const {
    if constexpr (kBMode == kMfmaF8BNative) {
      return qb[reg];
    } else {
      const uint32_t code = Planes::codes(qb, reg);
      if constexpr (kBMode == kMfmaF8BLut) {
        const uint32_t index = kHasZeroPoint ? code + zp_prepared[reg / kZpRegsPerBlock] : code;
        return mfma_lut_lookup<Traits::kLutBits>(index, lut);
      } else if constexpr (kHasZeroPoint) {
        return ((code | 0x80808080u) - zp_prepared[reg / kZpRegsPerBlock]) ^ 0x80808080u;
      } else {
        constexpr uint32_t kOffset = (0x80u - (1u << (kBits - 1))) * 0x01010101u;
        return (code + kOffset) ^ 0x80808080u;
      }
    }
  }

  // Final conversion of the integer accumulator (INT8 without flush scales).
  CUDA_INLINE void flush_int8() {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        PRAGMA_UNROLL
        for (uint32_t r = 0; r < 4; r++) regs_c[tile_m][tile_n][r] = static_cast<float>(regs_part[tile_m][tile_n][r]);
      }
    }
  }

  // A operand registers of MFMA `unit` of the chunk for one 16-row tile.
  CUDA_INLINE void load_a_unit(uint32_t stage, const mfma_u32x4_t *lds_buffer, uint32_t unit, uint32_t tile_m, uint32_t *dst) const {
    if constexpr (kLdsA) {
      const uint32_t row = local_m_offset + tile_m * 16 + lane % 16;
      const uint32_t lg = lane / 16;
      const uint8_t *row_ptr = reinterpret_cast<const uint8_t *>(lds_buffer + row * kRowUnits);
      auto unit_ptr = [&](uint32_t u) { return row_ptr + (u ^ row % kRowUnits) * 16; };
      if constexpr (kIsL32) {
        const uint32_t u = unit * 2 + lg / 2;
        const uint2 v = *reinterpret_cast<const uint2 *>(unit_ptr(u) + lg % 2 * 8);
        dst[0] = v.x, dst[1] = v.y;
      } else {
        // L128 (8-bit: units lg, 4 + lg) / TRUE (fp4: unit lg); int8 x64 unit u: unit 4 u + lg.
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kARegsPerUnit / 4; i++) {
          const uint32_t u = (kUnits == 2 ? unit : i) * 4 + lg;
          const mfma_u32x4_t v = *reinterpret_cast<const mfma_u32x4_t *>(unit_ptr(u));
          dst[i * 4 + 0] = v.x, dst[i * 4 + 1] = v.y, dst[i * 4 + 2] = v.z, dst[i * 4 + 3] = v.w;
        }
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kARegsPerUnit; i++) dst[i] = regs_a[stage][tile_m][unit * kARegsPerUnit + i];
    }
  }

  template <class AccT>
  CUDA_INLINE void mma(uint32_t stage, const uint32_t *a, const uint32_t *b, AccT &acc, uint32_t tile_m, uint32_t tile_n) const {
    if constexpr (kMfmaK == 128) {
      const uint32_t scale_a = kAsMode == kMfmaF8ScaleHw ? regs_asb[stage][tile_m] : 127u;
      const uint32_t scale_b = kWsMode == kMfmaF8ScaleHw ? regs_wsb[stage][tile_n] : 127u;
      mfma_scale_16x16x128<kAFmt, kBFmt>(a, b, acc, scale_a, scale_b);
    } else if constexpr (kMfmaK == 64) {
      mfma_i8_16x16x64(a, b, acc);
    } else if constexpr (kIsInt8) {
      mfma_i8_16x16x32(a, b, acc);
    } else {
      mfma_f8_16x16x32<kAFmt, kBFmt>(a, b, acc);
    }
  }

  CUDA_INLINE void compute_chunk(uint32_t stage, uint32_t chunk_k, const uint32_t *lut) {
    mfma_u32x4_t *lds_buffer = lds_a + chunk_k % 2 * kLdsBufferUnits;
    if constexpr (kLdsA) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLdsAUnits; i++) {
        const uint32_t unit_index = thread_id + i * TileConfig::kNumThreads;
        const uint32_t row = unit_index / kRowUnits;
        lds_buffer[row * kRowUnits + (unit_index % kRowUnits ^ row % kRowUnits)] = regs_a_share[stage][i];
      }
      if constexpr (kLdsAs) {
        float *lds_as_buffer = lds_as + chunk_k % 2 * kLdsAsWords;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kAsRowsPerThread; i++) {
          const uint32_t row = thread_id + i * TileConfig::kNumThreads;
          if (row < TileConfig::kBlockM) {
            PRAGMA_UNROLL
            for (uint32_t f = 0; f < kFlushesPerChunk; f++) lds_as_buffer[f * TileConfig::kBlockM + row] = __uint_as_float(regs_as_row[stage][i][f]);
          }
        }
      }
      __syncthreads();
    }

    uint32_t zp_prepared[kTilesN][kZpBlocks];
    if constexpr (kHasZeroPoint) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        PRAGMA_UNROLL
        for (uint32_t block = 0; block < kZpBlocks; block++) {
          const uint32_t zp = regs_bzp[stage][tile_n][block];
          // LUT index = code + (2^bits - 1 - zp) in [0, 2^(bits + 1) - 2]; INT8: subtract zp.
          zp_prepared[tile_n][block] = (kBMode == kMfmaF8BLut ? ((1u << kBits) - 1 - zp) : zp) * 0x01010101u;
        }
      }
    }

    // B is dequantized once per (flush group, tile_n) and reused by every tile_m; A is read from
    // LDS / registers per tile_m right before its MFMAs (keeps only one A tile live).
    PRAGMA_UNROLL
    for (uint32_t f = 0; f < kFlushesPerChunk; f++) {
      uint32_t b_flush[kTilesN][kUnitsPerFlush][kBRegsPerUnit];
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        PRAGMA_UNROLL
        for (uint32_t u = 0; u < kUnitsPerFlush; u++) {
          PRAGMA_UNROLL
          for (uint32_t i = 0; i < kBRegsPerUnit; i++) {
            b_flush[tile_n][u][i] = dequant_reg(regs_qb[stage][tile_n], (f * kUnitsPerFlush + u) * kBRegsPerUnit + i, lut, zp_prepared[tile_n]);
          }
        }
      }
      float ws[kTilesN];
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        if constexpr (kWsMode == kMfmaF8ScaleFlush) {
          ws[tile_n] = __uint_as_float(regs_ws[stage][tile_n][f]);
        } else if constexpr (kWsMode == kMfmaF8ScaleE8m0Flush) {
          ws[tile_n] = mfma_e8m0_to_float(regs_ws[stage][tile_n][f]);
        } else {
          ws[tile_n] = 1.0f;
        }
      }
      // A fragments and activation flush scales of one 16-row tile.
      auto load_a_tile = [&](uint32_t tile_m, uint32_t (&a_flush)[kUnitsPerFlush][kARegsPerUnit], float (&as_f)[4]) {
        PRAGMA_UNROLL
        for (uint32_t u = 0; u < kUnitsPerFlush; u++) load_a_unit(stage, lds_buffer, f * kUnitsPerFlush + u, tile_m, a_flush[u]);
        if constexpr (kLdsAs) {
          // C rows of this lane: (lane / 16) * 4 + r, r = 0..3 -> one 16-byte LDS read.
          const float *src = lds_as + chunk_k % 2 * kLdsAsWords + f * TileConfig::kBlockM + local_m_offset + tile_m * 16 + lane / 16 * 4;
          const float4 v = *reinterpret_cast<const float4 *>(src);
          as_f[0] = v.x, as_f[1] = v.y, as_f[2] = v.z, as_f[3] = v.w;
          return;
        }
        PRAGMA_UNROLL
        for (uint32_t r = 0; r < 4; r++) {
          if constexpr (kAsMode == kMfmaF8ScaleFlush) {
            as_f[r] = __uint_as_float(regs_as[stage][tile_m][f][r]);
          } else if constexpr (kAsMode == kMfmaF8ScaleE8m0Flush) {
            as_f[r] = mfma_e8m0_to_float(regs_as[stage][tile_m][f][r]);
          } else {
            as_f[r] = 1.0f;
          }
        }
      };
      if constexpr (kHasFlushScale) {
        // Every flush group (kScaleK <= 128) or chunk (kScaleK > 128: the group scale is constant
        // across its chunks) gets fresh accumulators, scaled into regs_c. The scaling of tile_m - 1
        // is issued after the MFMAs of tile_m so that it does not wait on the MFMA latency (up to
        // 8 tiles per wave; larger tiles run out of registers for the second set of accumulators).
        constexpr bool kDeferFlush = kTilesM * kTilesN <= 8;
        Acc acc_prev[kTilesN];
        float as_prev[4];
        auto scale_into_c = [&](uint32_t tile_m, Acc (&acc)[kTilesN], const float (&as_f)[4]) {
          PRAGMA_UNROLL
          for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
            PRAGMA_UNROLL
            for (uint32_t r = 0; r < 4; r++) {
              const float scale = kAsFlush ? ws[tile_n] * as_f[r] : ws[tile_n];
              regs_c[tile_m][tile_n][r] += static_cast<float>(acc[tile_n][r]) * scale;
            }
          }
        };
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          uint32_t a_flush[kUnitsPerFlush][kARegsPerUnit];
          float as_f[4];
          load_a_tile(tile_m, a_flush, as_f);
          if constexpr (!kDeferFlush) {
            // One live accumulator: scale each tile_n right after its MFMAs.
            PRAGMA_UNROLL
            for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
              Acc acc{};
              PRAGMA_UNROLL
              for (uint32_t u = 0; u < kUnitsPerFlush; u++) mma(stage, a_flush[u], b_flush[tile_n][u], acc, tile_m, tile_n);
              PRAGMA_UNROLL
              for (uint32_t r = 0; r < 4; r++) {
                const float scale = kAsFlush ? ws[tile_n] * as_f[r] : ws[tile_n];
                regs_c[tile_m][tile_n][r] += static_cast<float>(acc[r]) * scale;
              }
            }
            continue;
          }
          Acc acc[kTilesN];
          PRAGMA_UNROLL
          for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
            acc[tile_n] = Acc{};
            PRAGMA_UNROLL
            for (uint32_t u = 0; u < kUnitsPerFlush; u++) mma(stage, a_flush[u], b_flush[tile_n][u], acc[tile_n], tile_m, tile_n);
          }
          {
            if (tile_m > 0) scale_into_c(tile_m - 1, acc_prev, as_prev);
            PRAGMA_UNROLL
            for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) acc_prev[tile_n] = acc[tile_n];
            PRAGMA_UNROLL
            for (uint32_t r = 0; r < 4; r++) as_prev[r] = as_f[r];
          }
        }
        if constexpr (kDeferFlush) scale_into_c(kTilesM - 1, acc_prev, as_prev);
      } else {
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          uint32_t a_flush[kUnitsPerFlush][kARegsPerUnit];
          float as_f[4];
          load_a_tile(tile_m, a_flush, as_f);
          PRAGMA_UNROLL
          for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
            PRAGMA_UNROLL
            for (uint32_t u = 0; u < kUnitsPerFlush; u++) {
              if constexpr (kIsInt8) {
                // INT32 accumulator, converted at the end.
                mma(stage, a_flush[u], b_flush[tile_n][u], regs_part[tile_m][tile_n], tile_m, tile_n);
              } else {
                mma(stage, a_flush[u], b_flush[tile_n][u], regs_c[tile_m][tile_n], tile_m, tile_n);
              }
            }
          }
        }
      }
    }
  }

  CUDA_INLINE void mainloop(uint32_t chunk_begin, uint32_t chunk_end) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        regs_c[tile_m][tile_n] = mfma_f32x4_t{};
        if constexpr (kIsInt8 && !kHasFlushScale) regs_part[tile_m][tile_n] = Acc{};
      }
    }

    uint32_t lut[8];
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 8; i++) lut[i] = Traits::lut(i);

    const uint32_t num_iters = chunk_end - chunk_begin;
    if (num_iters == 0) return;
    const uint32_t last_chunk = chunk_end - 1;
    static_for<0, kStages - 1>([&](auto stage) {
      load_chunk(decltype(stage)::value, MIN(chunk_begin + decltype(stage)::value, last_chunk));
    });

    uint32_t base = 0;
    for (; base + kStages <= num_iters; base += kStages) {
      static_for<0, kStages>([&](auto stage_constant) {
        constexpr uint32_t stage = decltype(stage_constant)::value;
        constexpr uint32_t load_stage = (stage + kStages - 1) % kStages;
        load_chunk(load_stage, MIN(chunk_begin + base + stage + kStages - 1, last_chunk));
        __builtin_amdgcn_sched_barrier(0);
        compute_chunk(stage, chunk_begin + base + stage, lut);
      });
    }
    static_for<0, kStages - 1>([&](auto stage_constant) {
      constexpr uint32_t stage = decltype(stage_constant)::value;
      if (base + stage < num_iters) compute_chunk(stage, chunk_begin + base + stage, lut);
    });

    if constexpr (kIsInt8 && !kHasFlushScale) flush_int8();
  }

  template <class Func>
  CUDA_INLINE void for_each_output(Func &&func) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        if (!is_tile_n_valid(tile_n)) continue;
        PRAGMA_UNROLL
        for (uint32_t reg = 0; reg < 4; reg++) {
          const uint32_t row = m_offset + tile_m * 16 + lane / 16 * 4 + reg;
          if (row < shape_m) func(row, column(tile_n), regs_c[tile_m][tile_n][reg]);
        }
      }
    }
  }
};


template <class Traits, class TileConfig>
__global__ __launch_bounds__(TileConfig::kNumThreads, 2) void humming_mfma_f8(
    const uint8_t *__restrict__ A,
    const uint32_t *__restrict__ B,
    const void *__restrict__ WS,
    const uint8_t *__restrict__ BZP,
    const float *__restrict__ BS2,
    const void *__restrict__ AS,
    uint32_t as_stride,
    uint16_t *__restrict__ C,
    float *__restrict__ workspace,
    uint32_t *__restrict__ tile_counters,
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k) {
  using Gemm = MfmaF8Gemm<Traits, TileConfig>;
  using ElementC = typename Traits::ElementC;
  const uint32_t warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize);
  const uint32_t warp_m = warp_id / TileConfig::kWarpsN;
  const uint32_t warp_n = warp_id % TileConfig::kWarpsN;

  // Block -> output tile order, for L2 reuse: the hardware hands consecutive blocks to the 8 XCDs
  // in turn, so blocks on one XCD are first made contiguous, then grouped kGroupM M tiles at a
  // time (consecutive blocks share A rows and, across a group, B columns).
  constexpr uint32_t kXcds = 8;
  constexpr uint32_t kGroupM = 8;
  const uint32_t num_tiles_n = gridDim.x;
  const uint32_t num_tiles_m = gridDim.y;
  const uint32_t num_xy = num_tiles_n * num_tiles_m;
  uint32_t pid = blockIdx.y * num_tiles_n + blockIdx.x;
  if (num_xy % kXcds == 0) pid = pid % kXcds * (num_xy / kXcds) + pid / kXcds;
  const uint32_t group_first_m = pid / (kGroupM * num_tiles_n) * kGroupM;
  const uint32_t group_rows = MIN(num_tiles_m - group_first_m, kGroupM);
  const uint32_t block_tile_m = group_first_m + pid % (kGroupM * num_tiles_n) % group_rows;
  const uint32_t block_tile_n = pid % (kGroupM * num_tiles_n) / group_rows;

  __shared__ mfma_u32x4_t lds_a[TileConfig::kLdsA ? 2 * Gemm::kLdsBufferUnits : 1];
  __shared__ __attribute__((aligned(16))) float lds_as[Gemm::kLdsAs ? 2 * Gemm::kLdsAsWords : 1];
  Gemm gemm;
  gemm.lds_a = lds_a;
  gemm.lds_as = lds_as;
  gemm.thread_id = threadIdx.x;
  gemm.block_m_offset = block_tile_m * TileConfig::kBlockM;
  gemm.local_m_offset = warp_m * TileConfig::kWarpTilesM * 16;
  gemm.a_ptr = A;
  gemm.b_ptr = B;
  gemm.ws_ptr = WS;
  gemm.bzp_ptr = BZP;
  gemm.as_ptr = AS;
  gemm.as_stride = as_stride;
  gemm.shape_m = shape_m;
  gemm.shape_n = shape_n;
  gemm.shape_k = shape_k;
  gemm.lane = threadIdx.x % kWaveSize;
  gemm.m_offset = block_tile_m * TileConfig::kBlockM + warp_m * TileConfig::kWarpTilesM * 16;
  gemm.n_tile_offset = (block_tile_n * TileConfig::kBlockN + warp_n * TileConfig::kWarpTilesN * 16) / 16;

  const uint32_t num_chunks_k = shape_k / Gemm::kChunkK;
  const uint32_t num_splits = gridDim.z;
  const uint32_t chunk_begin = blockIdx.z * num_chunks_k / num_splits;
  const uint32_t chunk_end = (blockIdx.z + 1) * num_chunks_k / num_splits;
  gemm.mainloop(chunk_begin, chunk_end);

  // Epilogue: per-row activation scale (token / tensor) x per-column FP32 weight scale.
  auto store = [&](uint32_t row, uint32_t col, float value) {
    if constexpr (Traits::kAsMode == kMfmaF8ScaleNone) value *= static_cast<const float *>(AS)[row * as_stride];
    value *= BS2[col];
    C[static_cast<size_t>(row) * shape_n + col] = mfma_float_to_half<ElementC>(value);
  };

  if (num_splits == 1) {
    gemm.for_each_output(store);
    return;
  }

  // Split-K: identical to humming_mfma (see the comments there).
  constexpr uint32_t kTileFloats = TileConfig::kBlockM * TileConfig::kBlockN;
  constexpr uint32_t kSc1 = 16;
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

  constexpr uint32_t kRowsPerStep = MIN(Gemm::kTilesM, 4u);
  PRAGMA_UNROLL
  for (uint32_t tile_m_base = 0; tile_m_base < Gemm::kTilesM; tile_m_base += kRowsPerStep) {
    mfma_f32x4_t sums[kRowsPerStep][Gemm::kTilesN] = {};
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
  gemm.for_each_output(store);
  if (threadIdx.x == 0) tile_counters[tile_id] = 0;
}

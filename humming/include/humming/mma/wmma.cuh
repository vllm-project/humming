#pragma once

#include <humming/utils/all.cuh>


// Volta decode: of the four fp16x2 words a Turing lane would dequantize for 16-column group i, this lane needs the two
// of row half nh (words 2nh and 2nh+1). 4-bit: word w takes nibbles (w, w+4) of qb[i], so shift by 8nh and 8nh+4;
// 8-bit: word w takes bytes (w%2, w%2+2) of qb[2i + w/2], so qb[2i + nh] shifted by 0 and 8. No zero point.
template <class SourceType, class TargetType>
CUDA_INLINE void volta_dequant_half(const uint32_t *qb, uint32_t i, uint32_t nh, uint32_t *out) {
  static_assert(SourceType::kBits == 4 || SourceType::kBits == 8, "Volta decode path: int4 or int8 weights");
  static_assert(TargetType::kBits == 16, "Volta decode path: fp16 activations");
  const uint32_t zero = 0;
  if constexpr (SourceType::kBits == 4) {
    uint32_t w = qb[i] >> (nh * 8);
    out[0] = dequant_single<SourceType, TargetType, false, false>(w, zero);
    out[1] = dequant_single<SourceType, TargetType, false, false>(w >> 4, zero);
  } else {
    uint32_t w = nh ? qb[2 * i + 1] : qb[2 * i]; // a select, not a runtime register index (that spills)
    out[0] = dequant_single<SourceType, TargetType, false, false>(w, zero);
    out[1] = dequant_single<SourceType, TargetType, false, false>(w >> 8, zero);
  }
}


template <class Ctx, class ArithClass>
struct WMMA {
public:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementQB = typename Ctx::ElementQB;
  using ElementB = typename Ctx::ElementB;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using CRegistersArrayType = CRegistersType[WarpShape::M / MmaShape::M][WarpShape::N / MmaShape::N];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kNativeMixed = MmaOpClass::kNativeMixed;
  static constexpr bool kUseNativeDequant = Ctx::kUseNativeDequant && kUseNativeWeightDequant<ElementQB, ElementB>;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  Ctx &ctx;
  ArithClass &arith;
  typename MmaOpClass::ARegisters regs_a[2][WarpShape::M / MmaShape::M][kPartMmaShapeK / MmaShape::K];
  // Volta decode (volta.cuh): operands arrive in the m8n8k4 group layout; the weight loader brings two lanes' words.
#if HUMMING_VOLTA
  enum : bool { kVoltaDecode = Ctx::kVoltaDecode,
    kVoltaPrefill = Ctx::kVoltaPrefill };
  enum : uint32_t { kQbWords = ElementQB::kBits * (16 / ElementA::kBits) };
  uint32_t regs_qb[2][(Ctx::kVoltaNative ? 2 : 1) * kQbWords];
#else
  enum : bool { kVoltaDecode = false,
    kVoltaPrefill = false };
  enum : uint32_t { kQbWords = 0 };
  uint32_t regs_qb[2][ElementQB::kBits * (16 / ElementA::kBits)];
#endif
  typename MmaOpClass::BRegisters regs_b[2][WarpShape::N / MmaShape::N][kPartMmaShapeK / MmaShape::K];
  alignas(16) CRegistersArrayType regs_c[2];

  CUDA_INLINE
  WMMA(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
  }

  CUDA_INLINE
  void zero_accum() {
    uint32_t *regs_c_ptr = regs_c_as_ptr(0);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id, uint32_t iter_id) {
    if constexpr (Ctx::kUseLdmatrixDequant || (!kNativeMixed && std::is_same<ElementB, ElementQB>::value)) {
      return;
    } else if constexpr (kNativeMixed) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
        uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 16 / MmaShape::N]);
        repack_native_mxf8f6f4<ElementQB>(regs_qb[buffer_id], regs_b_ptr, i);
      }
    } else if constexpr (kUseFusedE8m0Scale) {
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
      fused_dequant_for_mxfp4<ElementB, WarpShape::N / 16, false>(regs_qb[buffer_id], regs_b_ptr, arith.bs[buffer_id]);
    } else if constexpr (kVoltaDecode) {
      static_assert(!Ctx::kUseLdmatrixDequant && !kNativeMixed && !kUseFusedE8m0Scale && kNumWarpShapeNSplits == 1,
          "Volta decode path: plain dequantized integer weights");
      const VoltaLane vl = volta_lane();
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
        uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 16 / MmaShape::N]);
        // [row 8nh+j: k 4kh..4kh+1 of step 0, of step 1; k 4kh+2..4kh+3 of step 0, of step 1]
        volta_dequant_half<ElementQB, ElementB>(regs_qb[buffer_id], i, vl.nh, regs_b_ptr);
        volta_dequant_half<ElementQB, ElementB>(regs_qb[buffer_id] + kQbWords, i, vl.nh, regs_b_ptr + 2);
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      }
    } else if constexpr (kVoltaPrefill) {
      static_assert(!Ctx::kUseLdmatrixDequant && !kNativeMixed && !kUseFusedE8m0Scale && kNumWarpShapeNSplits == 1,
          "Volta prompt path: plain dequantized integer weights");
      // Both source lanes' four words: tile 2i = column j (words 0, 1), tile 2i+1 = column 8+j (words 2, 3).
      // regs_b[2i] = {k0: src0 src1, k1: src0 src1}, regs_b[2i+1] the same for words 2, 3.
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
        uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 16 / MmaShape::N]);
        uint32_t words0[4], words1[4];
        dequant<ElementQB, ElementB, false, false, 1>(regs_qb[buffer_id], words0, i, nullptr);
        dequant<ElementQB, ElementB, false, false, 1>(regs_qb[buffer_id] + kQbWords, words1, i, nullptr);
        regs_b_ptr[0] = words0[0];
        regs_b_ptr[1] = words1[0];
        regs_b_ptr[2] = words0[1];
        regs_b_ptr[3] = words1[1];
        regs_b_ptr[4] = words0[2];
        regs_b_ptr[5] = words1[2];
        regs_b_ptr[6] = words0[3];
        regs_b_ptr[7] = words1[3];
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      }
    } else {
      if constexpr (ElementQB::kBits == 1 && kNumWarpShapeNSplits == 2) {
        regs_qb[buffer_id][0] = regs_qb[buffer_id][0] >> (ctx.warp_id() % 2 * (ElementA::kBits / 2));
      }

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < WarpShape::N / 16; i++) {
        uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i * 16 / MmaShape::N]);
        uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, i);
        uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
        if constexpr (kUseNativeDequant) {
          dequant_native<ElementQB, ElementB>(regs_qb[buffer_id], regs_b_ptr, i);
        } else {
          dequant<ElementQB, ElementB, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
        }
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      };
    }
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kPartMmaShapeK / MmaShape::K; k++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < WarpShape::N / MmaShape::N; j++) {
        PRAGMA_UNROLL
        for (uint32_t m = 0; m < WarpShape::M / MmaShape::M; m++) {
          if constexpr (ElementA::kBits == 16 && MmaShape::M == 8 && MmaShape::N == 16 && MmaShape::K == 8 && !USE_PPU) {
            uint32_t *weight_values = reinterpret_cast<uint32_t *>(regs_b[buffer_id][j]);
            uint32_t transposed_b[2] = {weight_values[k], weight_values[k + 2]};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
            // Volta: the second accumulator buffer holds this lane's other four outputs (volta.cuh).
            if constexpr (kVoltaDecode) {
              volta_mma_884_split(transposed_b, regs_a[buffer_id][m][k], regs_c[0][m][j], regs_c[1][m][j]);
            } else {
              volta_mma_m16n8k8_fast(transposed_b, regs_a[buffer_id][m][k], regs_c[0][m][j], regs_c[1][m][j]);
            }
#else
            MmaOpClass::fma(regs_a[buffer_id][m][k], transposed_b, regs_c[0][m][j], regs_c[0][m][j]);
#endif
          } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
            if constexpr (kVoltaPrefill) {
              volta_mma_884_split(regs_a[buffer_id][m][k], regs_b[buffer_id][j][k], regs_c[0][m][j], regs_c[1][m][j]);
            } else {
              volta_mma_m16n8k8_fast(regs_a[buffer_id][m][k], regs_b[buffer_id][j][k], regs_c[0][m][j], regs_c[1][m][j]);
            }
#else
            MmaOpClass::fma(regs_a[buffer_id][m][k], regs_b[buffer_id][j][k], regs_c[0][m][j], regs_c[0][m][j]);
#endif
          }
          arith.may_apply_as_and_bs_on_mma_c(regs_c_as_ptr(), m, j, k, iter_id);
        }
      }
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_a_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(regs_a[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    if constexpr (Ctx::kUseLdmatrixDequant || (!kNativeMixed && std::is_same<ElementB, ElementQB>::value)) {
      return reinterpret_cast<T *>(regs_b[buffer_id]);
    } else {
      return reinterpret_cast<T *>(regs_qb[buffer_id]);
    };
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_b_as_ptr() {
    return reinterpret_cast<T *>(regs_b);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    // Volta: add the two K halves and restore Turing's accumulator layout (volta.cuh). The second buffer
    // is free here: fp16 activations never select it (final_regs_c_index) and never scale C mid-loop.
    static_assert(ElementA::kBits == 16 && final_regs_c_index() == 0, "Volta path needs fp16 activations");
    PRAGMA_UNROLL
    for (uint32_t m = 0; m < WarpShape::M / MmaShape::M; m++) {
      PRAGMA_UNROLL
      for (uint32_t n = 0; n < WarpShape::N / MmaShape::N; n++) {
        volta_finalize_tile(regs_c[0][m][n], regs_c[1][m][n]);
      }
    }
#endif
    if constexpr (ElementA::kBits == 16 && MmaShape::M == 8 && MmaShape::N == 16 && !USE_PPU) {
      // Restore the logical 8x16 accumulator layout before the epilogue.
      uint32_t lane = ctx.lane_id();
      uint32_t source_lane = (lane % 4) * 8 + lane / 8;
      bool is_odd_row = (lane / 4) % 2;
      PRAGMA_UNROLL
      for (uint32_t m = 0; m < WarpShape::M / MmaShape::M; m++) {
        PRAGMA_UNROLL
        for (uint32_t n = 0; n < WarpShape::N / MmaShape::N; n++) {
          auto &values = regs_c[0][m][n];
          PRAGMA_UNROLL
          for (uint32_t column = 0; column < 2; column++) {
            if constexpr (sizeof(typename MmaOpClass::ValTypeC) == 4) {
              float even_first = __shfl_sync(0xffffffff, values[column * 2], source_lane);
              float odd_first = __shfl_sync(0xffffffff, values[column * 2 + 1], source_lane);
              float even_second = __shfl_sync(0xffffffff, values[column * 2], source_lane + 4);
              float odd_second = __shfl_sync(0xffffffff, values[column * 2 + 1], source_lane + 4);
              values[column * 2] = is_odd_row ? odd_first : even_first;
              values[column * 2 + 1] = is_odd_row ? odd_second : even_second;
            } else {
              uint32_t first = __shfl_sync(0xffffffff, values[column], source_lane);
              uint32_t second = __shfl_sync(0xffffffff, values[column], source_lane + 4);
              values[column] = __byte_perm(first, second, is_odd_row ? 0x7632 : 0x5410);
            }
          }
        }
      }
    }
    return regs_c_as_ptr<T>(final_regs_c_index());
  };

  static constexpr uint32_t final_regs_c_index() {
    uint32_t index = 0;
    constexpr bool kIsGroupInputScale = Ctx::kInputScaleGroupSize > 0;
    constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
    constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;

    if constexpr (ElementA::kBits < 16 && kIsGroupInputScale) {
      index = 1;
    }

    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && (kIsGroupWeightScale || kIsBlockWeightScale)) {
      index = 1;
    }

    return index;
  }
};

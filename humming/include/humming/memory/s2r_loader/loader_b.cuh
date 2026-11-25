#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class S2RMemoryLoaderB {
private:
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementQB = typename Ctx::ElementQB;

  static constexpr uint32_t N_WARPS = Ctx::N_WARPS;
  static constexpr uint32_t K_WARPS = Ctx::K_WARPS;

  static constexpr bool kIsWarpHalfGroup = WarpShape::N == ElementA::kBits * 2;
  static constexpr bool kLoadHalfGroup = ElementQB::kBits % 2 == 0 && kIsWarpHalfGroup && !Ctx::kUsePackedKLayout;
  static constexpr uint32_t TRUE_N_WARPS = kIsWarpHalfGroup ? N_WARPS / 2 : N_WARPS;
  static constexpr uint32_t kSmemStride = BlockShape::N * Ctx::kPartMmaShapeK * ElementQB::kBits / 32 / 4 * (Ctx::kUsePackedKLayout ? 2 : 1);
  static constexpr uint32_t kNumIntsPerThread = ElementQB::kBits / (kLoadHalfGroup ? 2 : 1);
  using LoadType = typename LoadTypeChooser<kNumIntsPerThread * 4>::Type;
  static constexpr uint32_t kLoadIters = kNumIntsPerThread / (sizeof(LoadType) / 4);

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderB(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load_packed_k(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    static_assert(Ctx::kPartMmaShapeK == 32);
    static_assert(ElementA::kBits == 8);
    static_assert(Ctx::kUseWgmma);
    static_assert(Ctx::kWarpIters == WarpShape::N / 16);
    static_assert(WarpShape::K == 128);

    uint32_t n_warp_id = ctx.n_warp_id();
    uint32_t lane_id = ctx.lane_id();

    constexpr uint32_t kKChunks = WarpShape::K / 64;
    constexpr uint32_t kPlane = N_WARPS * (WarpShape::N / 16) * 32;

    uint32_t idx = n_warp_id * (WarpShape::N / 16) + iter_id;
    idx = idx * 32 + lane_id;

    if constexpr (K_WARPS > 1) {
      idx += kPlane * kKChunks * ctx.k_warp_id();
    }

    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);

    PRAGMA_UNROLL
    for (uint32_t kc = 0; kc < kKChunks; kc++) {
      uint32_t smem_start_idx = (idx + kc * kPlane) * kLoadIters;
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLoadIters; j++) {
        reg_ptr_load[kc * kLoadIters + j] = smem_ptr_load[smem_start_idx + j];
      }
    }
  }

  CUDA_INLINE
  void load_native(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    if constexpr (USE_PPU && Ctx::kUseCpAsync && !Ctx::kUseTmaB)
      return load_raw_aiu(smem_ptr, regs_ptr, iter_id);
    constexpr uint32_t kSwizzleInt4s = MIN(8u, BlockShape::K * ElementQB::kBits / 128);
    uint32_t smem_base = cast_smem_ptr_to_uint(smem_ptr) / 128;
    uint32_t lane_id = ctx.lane_id();
    constexpr uint32_t kRowsPerLoad = USE_PPU ? 16 : 8;
    PRAGMA_UNROLL
    for (uint32_t n = 0; n < WarpShape::N / kRowsPerLoad; ++n) {
      uint32_t row = ctx.n_warp_offset() + n * kRowsPerLoad + lane_id % 8;
      if constexpr (USE_PPU) row += lane_id / 16 * 8;
      uint32_t col = (ctx.k_warp_offset() * ElementQB::kBits / 256 + iter_id) * 2 + (lane_id / 8) % 2;
      uint32_t offset = (col / kSwizzleInt4s * BlockShape::N + row) * kSwizzleInt4s + col % kSwizzleInt4s;
      uint32_t swizzled_offset = offset ^ ((smem_base + offset / 8) % kSwizzleInt4s);
      if constexpr (Ctx::kUseFlatWeight) swizzled_offset = col * BlockShape::N + row;
      ld_shared<kRowsPerLoad / 4>(smem_ptr + swizzled_offset, reinterpret_cast<int4 *>(regs_ptr + n * kRowsPerLoad / 4));
    }
  }

  CUDA_INLINE
  void load_raw_aiu(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    constexpr uint32_t kSwizzleInt4s = MIN(8u, BlockShape::K * ElementQB::kBits / 128);
    uint32_t col = (ctx.k_warp_offset() * ElementQB::kBits / 256 + iter_id) * 2;
    const int4 *slab_ptr = smem_ptr + col / kSwizzleInt4s * BlockShape::N * kSwizzleInt4s;
    PRAGMA_UNROLL
    for (uint32_t n = 0; n < WarpShape::N / 16; ++n) {
      aiu_ld_shared<4>(slab_ptr, regs_ptr + n * 4, BlockShape::N,
                       ctx.n_warp_offset() + n * 16, col % kSwizzleInt4s * sizeof(int4));
    }
  }

  CUDA_INLINE
  void load_dequant(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    constexpr uint32_t kSwizzleInt4s = MIN(8u, BlockShape::K * ElementQB::kBits / 128);
    constexpr uint32_t kRowsPerLoad = Ctx::kUseWgmma ? 16 : 8;
    uint32_t smem_base = cast_smem_ptr_to_uint(smem_ptr);
    uint32_t lane_id = ctx.lane_id();
    uint32_t col = ctx.k_warp_offset() * ElementQB::kBits / 128 + iter_id;
    uint32_t half = lane_id / kRowsPerLoad % 2;
    PRAGMA_UNROLL
    for (uint32_t n = 0; n < WarpShape::N / kRowsPerLoad; ++n) {
      uint32_t row = ctx.n_warp_offset() + n * kRowsPerLoad + lane_id % kRowsPerLoad;
      uint32_t offset = (col / kSwizzleInt4s * BlockShape::N + row) * kSwizzleInt4s + col % kSwizzleInt4s;
      offset ^= (smem_base / 128 + offset / 8) % kSwizzleInt4s;
      if constexpr (Ctx::kUseFlatWeight) offset = col * BlockShape::N + row;
      ld_shared_dequant<ElementQB::kBits, kRowsPerLoad / 4>(smem_base + offset * 16 + half * 8, regs_ptr + n * kRowsPerLoad / 4);
    }
  }

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    if constexpr (Ctx::kUseLdmatrixDequant) return load_dequant(smem_ptr, regs_ptr, iter_id);
    if constexpr (Ctx::kUseRawWeight || Ctx::kUseFlatWeight) return load_native(smem_ptr, regs_ptr, iter_id);
    if constexpr (Ctx::kUsePackedKLayout) return load_packed_k(smem_ptr, regs_ptr, iter_id);
    uint32_t warp_id = ctx.warp_id();
    uint32_t n_warp_id = ctx.n_warp_id();
    if (kIsWarpHalfGroup) n_warp_id = n_warp_id / 2;
    uint32_t lane_id = ctx.lane_id();
    constexpr uint32_t warp_weight_blocks = MAX(WarpShape::N / (ElementA::kBits * 4), 1);
    uint32_t idx = warp_weight_blocks * 32 * n_warp_id + lane_id;

    if constexpr (K_WARPS > 1) {
      idx = TRUE_N_WARPS * 32 * warp_weight_blocks * Ctx::kWarpIters * ctx.k_warp_id() + idx;
    }

    uint32_t smem_start_idx = idx * kLoadIters;
    smem_ptr = smem_ptr + kSmemStride * iter_id;
    const LoadType *smem_ptr_load = reinterpret_cast<const LoadType *>(smem_ptr);
    LoadType *reg_ptr_load = reinterpret_cast<LoadType *>(regs_ptr);
#if HUMMING_VOLTA
    if constexpr (Ctx::kVoltaNative) {
      // Volta: this lane's weight row lives in the packed words of two Turing lanes. Load both; transform_b
      // dequantizes them and keeps the half this lane's group covers. regs_qb holds twice the words.
      static_assert(!kLoadHalfGroup, "Volta decode path: no half-group loads");
      const VoltaLane vl = volta_lane();
      const uint32_t idx_base = idx - lane_id;
      PRAGMA_UNROLL
      for (uint32_t src = 0; src < 2; src++) {
        const uint32_t src_start_idx = (idx_base + (src ? vl.src1 : vl.src0)) * kLoadIters;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < warp_weight_blocks; i++) {
          PRAGMA_UNROLL
          for (uint32_t j = 0; j < kLoadIters; j++) {
            reg_ptr_load[(src * warp_weight_blocks + i) * kLoadIters + j] = smem_ptr_load[src_start_idx + 32 * kLoadIters * i + j];
          }
        }
      }
      return;
    }
#endif

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < warp_weight_blocks; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kLoadIters; j++) {
        if constexpr (kLoadHalfGroup) {
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[(smem_start_idx + 32 * kLoadIters * i) * 2 + warp_id % 2 * kLoadIters + j];
        } else {
          reg_ptr_load[i * kLoadIters + j] = smem_ptr_load[smem_start_idx + 32 * kLoadIters * i + j];
        }
      }
    }
  };
};

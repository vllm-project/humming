#pragma once

#include <humming/utils/all.cuh>


template <class Ctx>
class S2RMemoryLoaderA {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;

  Ctx &ctx;

public:
  CUDA_INLINE S2RMemoryLoaderA(Ctx &ctx) : ctx(ctx) {}

  CUDA_INLINE
  void load(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id, uint32_t stage_id = 0) {
    if constexpr (USE_PPU && !Ctx::kIsIndexedGemm && Ctx::kUseCpAsync && !Ctx::kUseTmaA)
      return load_aiu(smem_ptr, regs_ptr, iter_id);
    const uint32_t lane_id = ctx.lane_id();
    const uint32_t m_iter_id = ctx.m_warp_id();
    const uint32_t k_warp_id = ctx.k_warp_id();
    uint32_t smem_uint = offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
    uint32_t smem_base = smem_uint / 128 % (BlockShape::K * ElementA::kBits == 512 ? 4 : 8);

#if HUMMING_VOLTA
    if constexpr (Ctx::kVoltaPrefill) {
      // Volta prompts: in each 16-row m tile this lane supplies row 8mh + j, k 4kh..4kh+3 of each 8-k step.
      const VoltaLane vl = volta_lane();
      PRAGMA_UNROLL
      for (uint32_t load_iter_id = 0; load_iter_id < WarpShape::M / 16; load_iter_id++) {
        PRAGMA_UNROLL
        for (uint32_t s = 0; s < 2; s++) {
          uint32_t row = ctx.m_warp_offset() + load_iter_id * 16 + vl.nh * 8 + vl.j;
          uint32_t col = iter_id * 2 + k_warp_id * (Ctx::kWarpIters * 2) + s;
          if constexpr (BlockShape::K * ElementA::kBits > 1024) {
            row = BlockShape::M * (col / 8) + row;
            col = (col % 8) ^ ((row + smem_base) % 8);
          } else if constexpr (BlockShape::K * ElementA::kBits == 1024) {
            col = col ^ ((row + smem_base) % 8);
          } else if constexpr (BlockShape::K * ElementA::kBits == 512) {
            col = row % 2 * 4 + col;
            row = row / 2;
            col = col ^ ((row + smem_base) % 4);
          }
          uint32_t a_sh_rd = row * 8 + col;
          volta_ld_shared_v2(static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr + a_sh_rd)) + vl.kh * 8,
              regs_ptr + load_iter_id * 4 + s * 2);
        }
      }
      return;
    }
    if constexpr (Ctx::kVoltaDecode) {
      // Volta decode: this lane supplies activation row j (its group index) for k 4kh..4kh+3 of each 8-k step.
      static_assert(WarpShape::M == 8, "Volta decode path: an 8-row warp tile");
      const VoltaLane vl = volta_lane();
      PRAGMA_UNROLL
      for (uint32_t s = 0; s < 2; s++) {
        uint32_t row = ctx.m_warp_offset() + vl.j;
        uint32_t col = iter_id * 2 + k_warp_id * (Ctx::kWarpIters * 2) + s;
        if constexpr (BlockShape::K * ElementA::kBits > 1024) {
          row = BlockShape::M * (col / 8) + row;
          col = (col % 8) ^ ((row + smem_base) % 8);
        } else if constexpr (BlockShape::K * ElementA::kBits == 1024) {
          col = col ^ ((row + smem_base) % 8);
        } else if constexpr (BlockShape::K * ElementA::kBits == 512) {
          col = row % 2 * 4 + col;
          row = row / 2;
          col = col ^ ((row + smem_base) % 4);
        }
        uint32_t a_sh_rd = row * 8 + col;
        volta_ld_shared_v2(static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr + a_sh_rd)) + vl.kh * 8, regs_ptr + s * 2);
      }
      return;
    }
#endif
    PRAGMA_UNROLL
    for (uint32_t load_iter_id = 0; load_iter_id < CEIL_DIV(WarpShape::M, 16); load_iter_id++) {

      uint32_t row = ctx.m_warp_offset() + load_iter_id * 16;
      uint32_t col = iter_id * 2 + k_warp_id * (Ctx::kWarpIters * 2);

      if constexpr (MmaShape::M == 8 || USE_PPU) {
        row += (lane_id / 16) * 8 + lane_id % 8;
        col += (lane_id / 8) % 2;
      } else {
        row += lane_id % 16;
        col += lane_id / 16;
      }

      if constexpr (BlockShape::K * ElementA::kBits > 1024) {
        row = BlockShape::M * (col / 8) + row;
        col = (col % 8) ^ ((row + smem_base) % 8);
      } else if constexpr (BlockShape::K * ElementA::kBits == 1024) {
        col = col ^ ((row + smem_base) % 8);
      } else if constexpr (BlockShape::K * ElementA::kBits == 512) {
        col = row % 2 * 4 + col;
        row = row / 2;
        col = col ^ ((row + smem_base) % 4);
      }

      uint32_t a_sh_rd = row * 8 + col;

      if ((load_iter_id == CEIL_DIV(WarpShape::M, 16) - 1) && WarpShape::M % 16 == 8) {
        ld_shared<2>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr) + load_iter_id);
      } else {
        ld_shared<4>(smem_ptr + a_sh_rd, reinterpret_cast<int4 *>(regs_ptr) + load_iter_id);
      }
    };
  };

  CUDA_INLINE
  void load_aiu(const int4 *smem_ptr, uint32_t *regs_ptr, uint32_t iter_id) {
    const uint32_t k_warp_id = ctx.k_warp_id();
    uint32_t col = iter_id * 2 + k_warp_id * (Ctx::kWarpIters * 2);
    uint32_t subblock_id = col / 8;
    col = col % 8;

    uint32_t row = ctx.m_warp_offset();
    const int4 *smem_ptr_new = smem_ptr + subblock_id * (BlockShape::M * 1024 / 128);

    PRAGMA_UNROLL
    for (uint32_t load_iter_id = 0; load_iter_id < CEIL_DIV(WarpShape::M, 16); load_iter_id++) {
      aiu_ld_shared<4>(
          smem_ptr_new, regs_ptr + load_iter_id * 4,
          BlockShape::M,
          row, col * 16);
      row += 16;
    }
  }
};

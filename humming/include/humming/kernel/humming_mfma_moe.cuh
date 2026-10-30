#pragma once

// MoE (grouped) weight-only GEMM for AMD CDNA (gfx950), built on the dense MFMA kernel.
//
//   for every expert e:  C[rows(e), :n_out] = A[rows(e), :] @ dequant(B[e])[N, K]^T (+ bias[e])
//
// Reuses MfmaWeightOnlyGemm (humming_mfma.cuh) unchanged: its dequant, MFMA chunk math, group
// scales and register arrays. MfmaMoeGemm only replaces the A loads (rows are gathered through a
// per-block row map instead of being a contiguous M range) and the main loop that calls them;
// every M block belongs to exactly one expert, whose B / scale / zero-point bases are selected
// once per block.
//
// Row maps (the three MoE layouts of Humming, see humming/config GemmType):
//   indexed (1):            M block b covers sorted_ids[b * kBlockM, (b + 1) * kBlockM) of
//                           expert expert_ids[b] (moe_align_block_size with block = kBlockM).
//                           id < a_rows * top_k is valid: A row id / top_k, C row id.
//   grouped_contiguous (2): expert_layout = [E + 1] row offsets (int32 or int64); expert e owns
//                           rows [off[e], off[e + 1]) of A and C.
//   grouped_masked (3):     expert_layout = [E] row counts; expert e owns rows
//                           [e * max_tokens, e * max_tokens + count[e]) of A and C.
// For the grouped layouts gridDim.y is an upper bound of sum_e ceil(rows(e) / kBlockM); each block
// finds its (expert, block) with a block-wide scan over the expert row counts, surplus blocks exit.
//
// B: [E, N, K * bits / 32] (MFMA layout per expert), scales [E, groups, N], zero points
// [E, groups, N]. C rows are written for columns < n_out with row stride ldc (so the N padding
// of the transform never reaches the caller's buffer); the optional bias [E, bias_stride] is added
// in FP32 before rounding. Split-K reuses the dense kernel's sc1 partials + tile counters.

#include <humming/kernel/humming_mfma.cuh>

enum MfmaMoeType : uint32_t {
  kMfmaMoeIndexed = 1,
  kMfmaMoeGroupedContiguous = 2,
  kMfmaMoeGroupedMasked = 3,
};


template <class ElementA, class ElementQB, class TileConfig, uint32_t kGroupSize, bool kHasZeroPoint>
struct MfmaMoeGemm : MfmaWeightOnlyGemm<ElementA, ElementQB, TileConfig, kGroupSize, kHasZeroPoint> {
  using Base = MfmaWeightOnlyGemm<ElementA, ElementQB, TileConfig, kGroupSize, kHasZeroPoint>;
  using Op = typename Base::Op;
  using CRegisters = typename Base::CRegisters;
  static constexpr uint32_t kBits = Base::kBits;
  static constexpr uint32_t kTilesM = Base::kTilesM;
  static constexpr uint32_t kTilesN = Base::kTilesN;
  static constexpr uint32_t kStages = Base::kStages;
  static constexpr uint32_t kChunkK = Base::kChunkK;
  static constexpr uint32_t kMfmaPerChunk = Base::kMfmaPerChunk;
  static constexpr uint32_t kGroupsPerChunk = Base::kGroupsPerChunk;
  static constexpr bool kIsChannelScale = Base::kIsChannelScale;
  static constexpr bool kLdsA = Base::kLdsA;
  static constexpr bool kScaleOnC = Base::kScaleOnC;
  static constexpr uint32_t kLdsAUnits = Base::kLdsAUnits;
  // One gathered A row per LDS unit (kLdsA) or per 16-row tile (direct loads).
  static constexpr uint32_t kRowSlots = kLdsA ? kLdsAUnits : kTilesM;

  // Byte offset of the A row feeding each load slot; invalid rows point at a valid row (their
  // C rows are never stored), so the loads stay unconditional like the dense kernel's.
  uint32_t a_row_bytes[kRowSlots];

  // Same as MfmaWeightOnlyGemm::load_chunk except for the A addresses.
  CUDA_INLINE void load_chunk(uint32_t stage, uint32_t chunk_k) {
    const uint32_t num_chunks_k = this->shape_k / kChunkK;
    constexpr uint32_t kChunkBytesB = kWaveSize * kBits * 4;
    const auto rsrc_b = mfma_make_buffer(this->b_ptr + static_cast<size_t>(this->first_n_tile()) * num_chunks_k * kWaveSize * kBits);
    PRAGMA_UNROLL
    for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
      const uint32_t voffset = ((this->clamped_n_tile(tile_n) - this->first_n_tile()) * num_chunks_k * kWaveSize + this->lane) * kBits * 4;
      static_for<0, (kBits + 3) / 4>([&](auto part) {
        constexpr uint32_t i = decltype(part)::value * 4;
        mfma_buffer_load<MIN(kBits - i, 4)>(rsrc_b, voffset + i * 4, chunk_k * kChunkBytesB, &this->regs_qb[stage][tile_n][i]);
      });

      if constexpr (!kIsChannelScale) {
        const auto rsrc_bs = mfma_make_buffer(this->bs_ptr);
        PRAGMA_UNROLL
        for (uint32_t group = 0; group < kGroupsPerChunk; group++) {
          const uint32_t group_index = (chunk_k * kChunkK + group * MIN(kGroupSize, kChunkK)) / kGroupSize;
          const uint32_t column_offset = this->clamped_column(tile_n);
          this->regs_bs[stage][tile_n][group] =
              __builtin_amdgcn_raw_buffer_load_b16(rsrc_bs, column_offset * 2, group_index * this->shape_n * 2, 0);
          if constexpr (kHasZeroPoint) {
            const auto rsrc_bzp = mfma_make_buffer(this->bzp_ptr);
            this->regs_bzp[stage][tile_n][group] =
                __builtin_amdgcn_raw_buffer_load_b8(rsrc_bzp, column_offset, group_index * this->shape_n, 0);
          }
        }
      }
    }

    const auto rsrc_a = mfma_make_buffer(this->a_ptr);
    const uint32_t soffset_a = chunk_k * kChunkK * 2;
    if constexpr (kLdsA) {
      // Direct-to-LDS like the dense kernel: slot (wave_slot + lane) == thread_id + i * kNumThreads,
      // so a_row_bytes[i] is the row of this slot; the swizzle is applied on the global side.
      constexpr uint32_t kNumWaves = TileConfig::kNumThreads / kWaveSize;
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kLdsAUnits; i++) {
        const uint32_t wave_slot = (i * kNumWaves + this->warp_id) * kWaveSize;
        const uint32_t slot = wave_slot + this->lane;
        const uint32_t row = slot / 16;
        const uint32_t unit = slot % 16 ^ row % 16;
        mfma_buffer_load_lds_b128(rsrc_a, this->lds_a[stage] + wave_slot, a_row_bytes[i] + unit * 16, soffset_a);
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
        const uint32_t voffset = a_row_bytes[tile_m] + Op::ab_k(this->lane, 0) * 2;
        PRAGMA_UNROLL
        for (uint32_t mfma_index = 0; mfma_index < kMfmaPerChunk; mfma_index++) {
          this->regs_a[stage][tile_m][mfma_index] =
              __builtin_amdgcn_raw_buffer_load_b128(rsrc_a, voffset + mfma_index * Op::K * 2, soffset_a, 0);
        }
      }
    }
  }

  // Same as MfmaWeightOnlyGemm::mainloop, calling the gathering load_chunk above.
  CUDA_INLINE void mainloop(uint32_t chunk_begin, uint32_t chunk_end) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        this->regs_c[tile_m][tile_n] = CRegisters{};
        if constexpr (kScaleOnC) this->regs_group_c[tile_m][tile_n] = CRegisters{};
      }
    }

    uint32_t channel_zero_points[kTilesN];
    if constexpr (kHasZeroPoint && kIsChannelScale) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        channel_zero_points[tile_n] =
            this->is_tile_n_valid(tile_n) ? Base::Dequant::prepare_zero_point(this->bzp_ptr[this->column(tile_n)]) : 0;
      }
    }

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
        if constexpr (kLdsA) {
          mfma_wait_vmcnt<(kStages - 2) * Base::kLoadsPerChunk>();
          __builtin_amdgcn_s_barrier();
        }
        load_chunk(load_stage, MIN(chunk_begin + base + stage + kStages - 1, last_chunk));
        __builtin_amdgcn_sched_barrier(0);
        this->compute_chunk(stage, chunk_begin + base + stage, channel_zero_points);
      });
    }
    static_for<0, kStages - 1>([&](auto stage_constant) {
      constexpr uint32_t stage = decltype(stage_constant)::value;
      if (base + stage < num_iters) {
        if constexpr (kLdsA) {
          mfma_wait_vmcnt<(kStages - 2 - stage) * Base::kLoadsPerChunk>();
          __builtin_amdgcn_s_barrier();
        }
        this->compute_chunk(stage, chunk_begin + base + stage, channel_zero_points);
      }
    });

    if constexpr (kScaleOnC && kGroupSize > kChunkK) {
      if (num_iters > 0 && chunk_end * kChunkK % kGroupSize != 0) {
        PRAGMA_UNROLL
        for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
          const uint32_t group_index = (chunk_end - 1) * kChunkK / kGroupSize;
          this->regs_bs[0][tile_n][0] =
              this->is_tile_n_valid(tile_n) ? this->bs_ptr[static_cast<size_t>(group_index) * this->shape_n + this->column(tile_n)] : 0;
        }
        this->apply_group_scale(0, 0);
      }
    }

    if constexpr (kIsChannelScale) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        if (!this->is_tile_n_valid(tile_n)) continue;
        const float scale = mfma_half_to_float<ElementA>(this->bs_ptr[this->column(tile_n)]);
        PRAGMA_UNROLL
        for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
          PRAGMA_UNROLL
          for (uint32_t reg = 0; reg < Op::kRegsC; reg++) this->regs_c[tile_m][tile_n][reg] *= scale;
        }
      }
    }
  }

  // func(local_row, column, value) for every accumulator of a valid N tile; local_row is the
  // row inside the M block (the caller maps it through the C row map).
  template <class Func>
  CUDA_INLINE void for_each_block_output(Func &&func) {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < kTilesM; tile_m++) {
      PRAGMA_UNROLL
      for (uint32_t tile_n = 0; tile_n < kTilesN; tile_n++) {
        if (!this->is_tile_n_valid(tile_n)) continue;
        PRAGMA_UNROLL
        for (uint32_t reg = 0; reg < Op::kRegsC; reg++) {
          func(this->local_m_offset + tile_m * 16 + Op::c_row(this->lane, reg), this->column(tile_n), this->regs_c[tile_m][tile_n][reg]);
        }
      }
    }
  }
};


// Finds the (expert, first row, row count) of grouped M block `target`: experts are split into
// contiguous runs per thread, the per-thread block counts are prefix-summed in LDS and the owning
// thread walks its run. info[0] = -1 when `target` is past the last block.
template <uint32_t kNumThreads>
CUDA_INLINE void mfma_moe_locate_grouped_block(
    const void *expert_layout, bool is_int64, bool is_masked, uint32_t num_experts, uint32_t max_tokens,
    uint32_t block_m, uint32_t target, int32_t *info, uint32_t *scan) {
  auto load = [&](uint32_t index) -> int64_t {
    return is_int64 ? static_cast<const int64_t *>(expert_layout)[index] : static_cast<const int32_t *>(expert_layout)[index];
  };
  auto rows_of = [&](uint32_t expert) -> uint32_t {
    if (is_masked) return static_cast<uint32_t>(MIN(MAX(load(expert), int64_t(0)), static_cast<int64_t>(max_tokens)));
    return static_cast<uint32_t>(MAX(load(expert + 1) - load(expert), int64_t(0)));
  };
  auto first_row_of = [&](uint32_t expert) -> uint32_t {
    return is_masked ? expert * max_tokens : static_cast<uint32_t>(load(expert));
  };

  const uint32_t tid = threadIdx.x;
  const uint32_t per_thread = (num_experts + kNumThreads - 1) / kNumThreads;
  const uint32_t expert_begin = MIN(tid * per_thread, num_experts);
  const uint32_t expert_end = MIN(expert_begin + per_thread, num_experts);
  uint32_t local_blocks = 0;
  for (uint32_t e = expert_begin; e < expert_end; e++) local_blocks += (rows_of(e) + block_m - 1) / block_m;

  if (tid == 0) info[0] = -1;
  scan[tid] = local_blocks;
  __syncthreads();
  PRAGMA_UNROLL
  for (uint32_t offset = 1; offset < kNumThreads; offset *= 2) {
    const uint32_t value = tid >= offset ? scan[tid - offset] : 0;
    __syncthreads();
    scan[tid] += value;
    __syncthreads();
  }
  const uint32_t inclusive = scan[tid];
  uint32_t acc = inclusive - local_blocks;
  if (target >= acc && target < inclusive) {
    for (uint32_t e = expert_begin; e < expert_end; e++) {
      const uint32_t rows = rows_of(e);
      const uint32_t blocks = (rows + block_m - 1) / block_m;
      if (target < acc + blocks) {
        const uint32_t local_block = target - acc;
        info[0] = static_cast<int32_t>(e);
        info[1] = static_cast<int32_t>(first_row_of(e) + local_block * block_m);
        info[2] = static_cast<int32_t>(MIN(block_m, rows - local_block * block_m));
        break;
      }
      acc += blocks;
    }
  }
  __syncthreads();
}


template <class ElementA, class ElementQB, class TileConfig, uint32_t kGroupSize, bool kHasZeroPoint, uint32_t kMoeType>
__global__ __launch_bounds__(TileConfig::kNumThreads, 2) void humming_mfma_moe(
    const uint16_t *__restrict__ A,
    const uint32_t *__restrict__ B,
    const uint16_t *__restrict__ BS,
    const uint8_t *__restrict__ BZP,
    const uint16_t *__restrict__ bias,
    uint16_t *__restrict__ C,
    float *__restrict__ workspace,
    uint32_t *__restrict__ tile_counters,
    const int32_t *__restrict__ sorted_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_padded,
    const void *__restrict__ expert_layout,
    uint32_t expert_layout_is_int64,
    uint32_t a_rows,
    uint32_t top_k,
    uint32_t num_experts,
    uint32_t shape_n,
    uint32_t shape_k,
    uint32_t n_out,
    uint32_t ldc,
    uint32_t bias_stride) {
  using Gemm = MfmaMoeGemm<ElementA, ElementQB, TileConfig, kGroupSize, kHasZeroPoint>;
  static_assert(kMoeType >= kMfmaMoeIndexed && kMoeType <= kMfmaMoeGroupedMasked);
  constexpr uint32_t kBlockM = TileConfig::kBlockM;
  constexpr uint32_t kNumThreads = TileConfig::kNumThreads;
  constexpr uint32_t kInvalidRow = 0xFFFFFFFFu;

  constexpr uint32_t kLdsABufferUnits = TileConfig::kLdsA ? kBlockM * 16 : 1;
  __shared__ mfma_u32x4_t lds_a0[kLdsABufferUnits];
  __shared__ mfma_u32x4_t lds_a1[TileConfig::kStages > 1 ? kLdsABufferUnits : 1];
  __shared__ mfma_u32x4_t lds_a2[TileConfig::kStages > 2 ? kLdsABufferUnits : 1];
  __shared__ mfma_u32x4_t lds_a3[TileConfig::kStages > 3 ? kLdsABufferUnits : 1];
  static_assert(TileConfig::kStages <= 4);
  __shared__ uint32_t lds_a_rows[kBlockM];
  __shared__ uint32_t lds_c_rows[kBlockM];
  __shared__ int32_t lds_block_info[3];

  // ---- which expert / rows this M block covers ------------------------------------------------
  int32_t expert;
  if constexpr (kMoeType == kMfmaMoeIndexed) {
    const uint32_t block_row = blockIdx.y * kBlockM;
    if (static_cast<int64_t>(block_row) >= static_cast<int64_t>(num_tokens_padded[0])) return;
    expert = __builtin_amdgcn_readfirstlane(expert_ids[blockIdx.y]);
    if (expert < 0 || static_cast<uint32_t>(expert) >= num_experts) return;
    const uint32_t num_valid = a_rows * top_k;
    for (uint32_t r = threadIdx.x; r < kBlockM; r += kNumThreads) {
      const uint32_t id = static_cast<uint32_t>(sorted_ids[block_row + r]);
      const bool valid = id < num_valid;
      lds_a_rows[r] = valid ? id / top_k : 0;
      lds_c_rows[r] = valid ? id : kInvalidRow;
    }
  } else {
    __shared__ uint32_t lds_scan[kNumThreads];
    constexpr bool kMasked = kMoeType == kMfmaMoeGroupedMasked;
    const uint32_t max_tokens = kMasked ? a_rows / num_experts : 0;
    mfma_moe_locate_grouped_block<kNumThreads>(
        expert_layout, expert_layout_is_int64 != 0, kMasked, num_experts, max_tokens, kBlockM, blockIdx.y,
        lds_block_info, lds_scan);
    expert = __builtin_amdgcn_readfirstlane(lds_block_info[0]);
    if (expert < 0) return;
    const uint32_t first_row = static_cast<uint32_t>(lds_block_info[1]);
    const uint32_t num_rows = static_cast<uint32_t>(lds_block_info[2]);
    for (uint32_t r = threadIdx.x; r < kBlockM; r += kNumThreads) {
      const uint32_t row = first_row + r;
      const bool valid = r < num_rows && row < a_rows;
      lds_a_rows[r] = valid ? row : MIN(first_row, a_rows - 1);
      lds_c_rows[r] = valid ? row : kInvalidRow;
    }
  }
  __syncthreads();

  // ---- the dense GEMM state, pointed at this expert -------------------------------------------
  const uint32_t warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize);
  const uint32_t warp_m = warp_id / TileConfig::kWarpsN;
  const uint32_t warp_n = warp_id % TileConfig::kWarpsN;
  const uint32_t num_groups = kGroupSize == 0 ? 1 : shape_k / kGroupSize;
  const size_t expert_index = static_cast<uint32_t>(expert);

  Gemm gemm;
  mfma_u32x4_t *lds_buffers[4] = {lds_a0, lds_a1, lds_a2, lds_a3};
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < TileConfig::kStages; i++) gemm.lds_a[i] = lds_buffers[i];
  gemm.warp_id = warp_id;
  gemm.thread_id = threadIdx.x;
  gemm.block_m_offset = 0;
  gemm.local_m_offset = warp_m * TileConfig::kWarpTilesM * 16;
  gemm.a_ptr = A;
  gemm.b_ptr = B + expert_index * shape_n * (shape_k / 32 * Gemm::kBits);
  gemm.bs_ptr = BS + expert_index * num_groups * shape_n;
  gemm.bzp_ptr = kHasZeroPoint ? BZP + expert_index * num_groups * shape_n : BZP;
  gemm.shape_m = kBlockM;
  gemm.shape_n = shape_n;
  gemm.shape_k = shape_k;
  gemm.lane = threadIdx.x % kWaveSize;
  gemm.m_offset = gemm.local_m_offset;
  gemm.n_tile_offset = (blockIdx.x * TileConfig::kBlockN + warp_n * TileConfig::kWarpTilesN * 16) / 16;

  if constexpr (Gemm::kLdsA) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < Gemm::kRowSlots; i++) {
      const uint32_t unit_index = threadIdx.x + i * kNumThreads;
      gemm.a_row_bytes[i] = lds_a_rows[unit_index / 16] * shape_k * 2;
    }
  } else {
    PRAGMA_UNROLL
    for (uint32_t tile_m = 0; tile_m < Gemm::kRowSlots; tile_m++) {
      gemm.a_row_bytes[tile_m] = lds_a_rows[gemm.local_m_offset + tile_m * 16 + Gemm::Op::a_row(gemm.lane)] * shape_k * 2;
    }
  }

  const uint32_t num_chunks_k = shape_k / Gemm::kChunkK;
  const uint32_t num_splits = gridDim.z;
  const uint32_t chunk_begin = blockIdx.z * num_chunks_k / num_splits;
  const uint32_t chunk_end = (blockIdx.z + 1) * num_chunks_k / num_splits;
  gemm.mainloop(chunk_begin, chunk_end);

  const uint16_t *expert_bias = bias == nullptr ? nullptr : bias + expert_index * bias_stride;
  auto store_outputs = [&]() {
    gemm.for_each_block_output([&](uint32_t local_row, uint32_t col, float value) {
      const uint32_t row = lds_c_rows[local_row];
      if (row == kInvalidRow || col >= n_out) return;
      if (expert_bias != nullptr) value += mfma_half_to_float<ElementA>(expert_bias[col]);
      C[static_cast<size_t>(row) * ldc + col] = mfma_float_to_half<ElementA>(value);
    });
  };

  if (num_splits == 1) {
    store_outputs();
    return;
  }

  // Split-K: identical to humming_mfma (sc1 partials in the fragment layout, last block reduces).
  constexpr uint32_t kTileFloats = TileConfig::kBlockM * TileConfig::kBlockN;
  constexpr uint32_t kSc1 = 16;
  const uint32_t tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  const uint32_t num_tiles = gridDim.x * gridDim.y;
  auto fragment_offset = [&](uint32_t tile_m, uint32_t tile_n) {
    return ((tile_m * Gemm::kTilesN + tile_n) * kNumThreads + threadIdx.x) * 16;
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
  store_outputs();
  if (threadIdx.x == 0) tile_counters[tile_id] = 0;
}

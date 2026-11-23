#pragma once

#include <humming/kernel/process_input/activation.cuh>
#include <humming/kernel/process_input/quantization.cuh>
#include <humming/utils/all.cuh>
#include <humming/utils/ptx/tcgen05.cuh>


CUDA_INLINE void shlf_trans_mma_c_32b(void *vals_ptr) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(vals_ptr);

  uint32_t val;
  uint32_t idx = (threadIdx.x / 4) % 2;
  switch (idx) {
    case 0: {
      val = vals_uint_ptr[1];
      break;
    };
    case 1: {
      val = vals_uint_ptr[0];
      break;
    }
  }

  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);

  switch (idx) {
    case 0: {
      vals_uint_ptr[1] = swapped_val;
      break;
    };
    case 1: {
      vals_uint_ptr[0] = swapped_val;
      break;
    }
  }
}


CUDA_INLINE void shlf_trans_mma_c_16b(void *vals_ptr) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(vals_ptr);

  uint32_t &val = vals_uint_ptr[0];
  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);
  uint32_t idx = (threadIdx.x / 4) % 2;

  uint16_t *vals_ushort_ptr = reinterpret_cast<uint16_t *>(&val);
  uint16_t *swapped_vals_ushort_ptr = reinterpret_cast<uint16_t *>(&swapped_val);

  switch (idx) {
    case 0: {
      vals_ushort_ptr[1] = swapped_vals_ushort_ptr[0];
      break;
    };
    case 1: {
      vals_ushort_ptr[0] = swapped_vals_ushort_ptr[1];
      break;
    }
  }
}


template <typename T>
CUDA_INLINE void shlf_trans_mma_c(T &vals) {
  static_assert(sizeof(T) == 8 || sizeof(T) == 4);
  if constexpr (sizeof(T) == 8) {
    shlf_trans_mma_c_32b(&vals);
  } else {
    shlf_trans_mma_c_16b(&vals);
  }
}


template <class Ctx, class MMA, class ArithClass>
class EpilogueSmemWriter : F16Conversion<typename Ctx::ElementC> {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementC = typename Ctx::ElementC;
  static constexpr bool kIsBinaryActivation = Ctx::kActivationFuncType == ActivationType::BinaryInterleaved;

  static constexpr bool kUseWgmma = Ctx::kUseWgmma;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;
  using MmaShape = typename MmaOpClass::MmaShape;
  using ValTypeC = typename MmaOpClass::ValTypeC;
  using CRegistersArrayType = typename MMA::CRegistersArrayType;

  static constexpr uint32_t kOutputRows = SharedStorage::kOutputRows;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;
  static constexpr bool kApplyUmmaActivationInRegisters = Ctx::kHasFusedOutput && !Ctx::kRequantizeOutput;
  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kUseIntWeightScale = Ctx::kUseIntWeightScale;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kHasGroupScale = kIsGroupInputScale || kIsGroupWeightScale || kIsBlockWeightScale;
  static constexpr bool kIsIntAccum = std::is_same<ValTypeC, int32_t>::value && (!kHasGroupScale || kUseIntWeightScale || (kUseFusedE8m0Scale && !kIsGroupInputScale));

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

public:
  Ctx &ctx;
  ArithClass &arith;

  CUDA_INLINE
  EpilogueSmemWriter(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
  }

  CUDA_INLINE
  void write(uint32_t *regs_ptr, uint32_t first_row) {
    if (ctx.k_warp_id() != 0) return;

    auto &regs = *reinterpret_cast<CRegistersArrayType *>(regs_ptr);
    scalar_t2 *smem_half2_ptr = reinterpret_cast<scalar_t2 *>(ctx.smem.reduce);
    uint32_t smem = offsetof(SharedStorage, reduce) / 128 % 8;
    using PackTypeC = std::conditional_t<
        sizeof(ValTypeC) == 2, scalar_t2,
        std::conditional_t<kIsIntAccum, int2, float2>>;

    uint32_t laneid = ctx.lane_id();
    uint32_t warpid = ctx.warp_id();
    uint32_t warp_delta_row = ctx.m_warp_offset();
    uint32_t n_warp_id = ctx.n_warp_id();
    uint32_t group_warp_id = warpid % 4;
    auto write_to_smem = [&](PackTypeC val, uint32_t row_8x8block, uint32_t col_8x8block) {
      scalar_t2 val_half2;

      if constexpr (Ctx::kOutputChunkRows) {
        uint32_t output_row = warp_delta_row + 8 * (kUseWgmma ? col_8x8block : row_8x8block);
        if (output_row < first_row || output_row >= first_row + kOutputRows) return;
      }

      if constexpr (kUseWgmma) shlf_trans_mma_c(val);
      if constexpr (USE_PPU && sizeof(ValTypeC) == 2) {
        // PPU use_f16_accum requires an additional accumulator layout conversion.
        uint32_t &bits = *reinterpret_cast<uint32_t *>(&val);
        uint32_t source_lane = (laneid >> 1) & 1u;
        uint32_t even = __shfl_sync(0xffffffff, bits, source_lane, 4);
        uint32_t odd = __shfl_sync(0xffffffff, bits, source_lane + 2, 4);
        bits = __byte_perm(even, odd, (laneid & 1u) ? 0x7632 : 0x5410);
      }
      if constexpr (sizeof(ValTypeC) != 4) {
        val_half2 = val;
      } else if constexpr (kIsIntAccum) {
        float2 val_float2 = {__int2float_rn(val.x), __int2float_rn(val.y)};
        if constexpr (kUseWgmma) {
          arith.may_apply_f32_on_smem_write(val_float2, col_8x8block, row_8x8block);
        } else {
          arith.may_apply_f32_on_smem_write(val_float2, row_8x8block, col_8x8block);
        }
        val_half2 = this->float22num2(val_float2);
      } else {
        if constexpr (kUseWgmma) {
          arith.may_apply_f32_on_smem_write(val, col_8x8block, row_8x8block);
        } else {
          arith.may_apply_f32_on_smem_write(val, row_8x8block, col_8x8block);
        }
        val_half2 = this->float22num2(val);
      };

      uint32_t &val_uint = *reinterpret_cast<uint32_t *>(&val_half2);
      if constexpr (kUseWgmma) {
        arith.may_apply_on_smem_write(val_uint, col_8x8block, row_8x8block);
      } else {
        arith.may_apply_on_smem_write(val_uint, row_8x8block, col_8x8block);
      }

      apply_output_activation(val_uint);

      if constexpr (!kUseWgmma) {
        uint32_t sub_row = laneid / 4;
        uint32_t row = warp_delta_row + 8 * row_8x8block + sub_row - first_row;
        uint32_t col = col_8x8block * 4 + WarpShape::N / 2 * n_warp_id;

        row = row + kOutputRows * (col / 32);
        col = ((col % 32 / 4) ^ ((sub_row + smem) % 8)) * 4 + laneid % 4;

        uint32_t idx = row * 32 + col;
        smem_half2_ptr[idx] = val_half2;
      } else {
        uint32_t sub_row = (laneid % 4) * 2 + (laneid % 8) / 4;
        uint32_t row = warp_delta_row + 8 * col_8x8block + sub_row - first_row;

        uint32_t output_warp = n_warp_id;
        if constexpr (Ctx::kUseWgmmaSsNLayout) {
          output_warp = ctx.wgmma_ss_n_tile(row_8x8block / 2);
          row_8x8block %= 2;
        }
        constexpr uint32_t count = Ctx::kUseWgmmaSsNLayout ? 4 : (64 / WarpShape::N);
        uint32_t col1 = ((output_warp % count * (8 / count) + row_8x8block) ^ ((sub_row + smem) % 8)) * 4 + laneid / 8;
        uint32_t col2 = (output_warp / count) * (kOutputRows * 64 / 2);
        uint32_t idx = row * 32 + col1 + col2;
        smem_half2_ptr[idx] = val_half2;
      }
    };

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs) / sizeof(regs[0]); i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < sizeof(regs[0]) / sizeof(regs[0][0]); j++) {
        auto part_regs = reinterpret_cast<PackTypeC *>(&regs[i][j]);
        constexpr uint32_t inner_m = (kUseWgmma ? (MmaShape::N / 4) : MmaShape::M) / 8;
        constexpr uint32_t inner_n = sizeof(regs[0][0]) / sizeof(PackTypeC) / inner_m;

        PRAGMA_UNROLL
        for (uint32_t m = 0; m < inner_m; m++) {
          PRAGMA_UNROLL
          for (uint32_t n = 0; n < inner_n; n++) {
            uint32_t row_index = i * inner_m + (USE_PPU ? n : m);
            uint32_t col_index = j * inner_n + (USE_PPU ? m : n);
            write_to_smem(part_regs[n * inner_m + m], row_index, col_index);
          }
        }
      }
    }
  }

  template <uint32_t kColumns, class GmemWriter>
  CUDA_INLINE void prepare_output(GmemWriter &writer, uint32_t first_row, uint32_t rows, uint32_t buffer_offset, uint32_t first_column = 0) {
    constexpr bool kUseTmaC = Ctx::kUseTmaC;
    constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;
    constexpr uint32_t kOutputWidth = GmemWriter::kOutputWidth;
    using Activation = typename Ctx::LayerConfig::ActivationFunc;
    using OutputType = typename Ctx::LayerConfig::OutputType;
    using ScaleType = typename Ctx::LayerConfig::OutputScaleType;
    constexpr uint32_t kDivisor = (kIsBinaryActivation ? 2 : 1);
    constexpr uint32_t kColumnsOut = kColumns / kDivisor;
    constexpr uint32_t kValues = []() {
      if constexpr (Ctx::kRequantizeOutput) {
        // Smaller vectors keep math threads active for narrow output tiles.
        if constexpr (Ctx::kUseMxmma && !kUseTmaC && Ctx::kInputScaleGroupSize <= 128 && kOutputRows * kColumnsOut < kNumMathThreads * 8) return 4u;
        return kOutputRows * kColumnsOut < kNumMathThreads * 16 ? 8u : 16u;
      }
      if constexpr (!kUseTmaC && kOutputWidth % 8 == 0) return 8u;
      return 2u;
    }();
    constexpr uint32_t kBits = Ctx::kOutputBits;
    constexpr uint32_t kStoreBytes = SharedStorage::kOutputTileColumns / kDivisor * kBits / 8;
    constexpr uint32_t kGroupSize = Ctx::kInputScaleGroupSize;
    uint32_t thread = ctx.math_thread_id();
    uint32_t smem_base = offsetof(SharedStorage, reduce) / 128 % 8;
    // Callers have already synchronized the shared output. Only TMA needs another
    // barrier after each issuing thread drains its previous stores.
    if constexpr (kUseTmaC) {
      tma_wait_store_group<0, true>();
      ctx.sync_math_threads();
    }
    constexpr uint32_t kIterations = CEIL_DIV(kOutputRows * kColumnsOut, kNumMathThreads * kValues);
    scalar_t staged[kUseTmaC ? kIterations * kValues * kDivisor : 1];
    if constexpr (kUseTmaC) {
      // Preserve this chunk before changing its layout in place for TMA.
      PRAGMA_UNROLL
      for (uint32_t iter = 0; iter < kIterations; iter++) {
        uint32_t index = (thread + iter * kNumMathThreads) * kValues;
        if (index >= kOutputRows * kColumnsOut) continue;
        uint32_t row = index / kColumnsOut;
        uint32_t column = index % kColumnsOut;
        PRAGMA_UNROLL
        for (uint32_t item = 0; item < kValues * kDivisor; item++) {
          uint32_t source_col = first_column + column * kDivisor + item;
          uint32_t source_row = source_col / 64 * kOutputRows + row;
          uint32_t word = (source_col % 64 / 8) ^ ((source_row + smem_base) % 8);
          uint32_t offset = (buffer_offset + source_row * 8 + word) * 8 + source_col % 8;
          staged[iter * kValues * kDivisor + item] = reinterpret_cast<const scalar_t *>(ctx.smem.reduce)[offset];
        }
      }
      ctx.sync_math_threads();
    }
    auto output_smem = reinterpret_cast<uint8_t *>(ctx.smem.reduce + buffer_offset);
    constexpr uint32_t kOutputUnroll = kUseTmaC ? kIterations : 1;
    PRAGMA_UNROLL_COUNT(kOutputUnroll)
    for (uint32_t index = thread * kValues; index < kOutputRows * kColumnsOut; index += kNumMathThreads * kValues) {
      uint32_t row = index / kColumnsOut;
      uint32_t column = index % kColumnsOut;
      uint32_t output_row = first_row + row;
      if constexpr (kIsIndexedGemm) output_row = writer.get_output_row_index(output_row);
      else output_row += writer.row_offset;
      uint32_t output_column = (writer.col_offset + first_column) / kDivisor + column;
      bool valid = row < rows && output_row < writer.output_shape_m && output_column < kOutputWidth;
      float values[kValues] = {};
      float source_values[!kUseTmaC ? kValues * kDivisor : 1] = {};
      if constexpr (!kUseTmaC) {
        constexpr uint32_t kLoadValues = MIN(8u, kValues * kDivisor);
        PRAGMA_UNROLL
        for (uint32_t item = 0; item < kValues * kDivisor; item += kLoadValues) {
          uint32_t source_col = first_column + column * kDivisor + item;
          uint32_t source_row = source_col / 64 * kOutputRows + row;
          uint32_t word = (source_col % 64 / 8) ^ ((source_row + smem_base) % 8);
          uint32_t offset = (buffer_offset + source_row * 8 + word) * 8 + source_col % 8;
          auto source = reinterpret_cast<const scalar_t *>(ctx.smem.reduce) + offset;
          load_values<scalar_t, kLoadValues>(source_values + item, source);
        }
      }
      PRAGMA_UNROLL
      for (uint32_t item = 0; item < kValues; item++) {
        float pair[kDivisor];
        PRAGMA_UNROLL
        for (uint32_t part = 0; part < kDivisor; part++) {
          if constexpr (!kUseTmaC) {
            pair[part] = source_values[item * kDivisor + part];
          } else {
            scalar_t value = staged[index / (kNumMathThreads * kValues) * kValues * kDivisor + item * kDivisor + part];
            pair[part] = static_cast<float>(value);
          }
        }
        if constexpr (!Ctx::kRequantizeOutput) values[item] = pair[0];
        else if constexpr (kIsBinaryActivation) values[item] = Activation::apply(pair[0], pair[1]);
        else if constexpr (Activation::kType == ActivationType::Unary) values[item] = Activation::apply(pair[0]);
        else values[item] = pair[0];
        if (!valid) values[item] = 0.f;
      }
      uint8_t *destination = nullptr;
      if constexpr (kUseTmaC) {
        uint32_t byte = column * kBits / 8;
        destination = output_smem + (byte / kStoreBytes * kOutputRows + row) * kStoreBytes + byte % kStoreBytes;
      }
      if constexpr (Ctx::kRequantizeOutput) {
        auto result = quant_dynamic_group<OutputType, ScaleType, kValues, kGroupSize, kNumMathThreads / 32>(values, nullptr);
        if constexpr (kUseTmaC) store_packed<kValues * kBits / 8>(destination, result.packed, !valid);
        else if (valid) writer.template store_output_packed<kValues * kBits / 8>(output_row, output_column, result.packed);
        if (valid && output_column % kGroupSize == 0) {
          constexpr uint32_t kGroups = CEIL_DIV(kOutputWidth, kGroupSize);
          constexpr bool kPackedScale = !std::is_same<ScaleType, Float32>::value;
          constexpr uint32_t kScaleColumns = kPackedScale ? CEIL_DIV(kGroups, 4) * 4 : kGroups;
          uint64_t scale_index;
          uint32_t group = output_column / kGroupSize;
          uint64_t scale_rows = CEIL_DIV(ctx.params.shape_m * (kIsIndexedGemm ? ctx.params.top_k : 1), 4) * 4;
          if constexpr (!Ctx::kUseMMajorInputScale) scale_index = uint64_t(output_row) * kScaleColumns + group;
          else if constexpr (!kPackedScale) scale_index = uint64_t(group) * scale_rows + output_row;
          else scale_index = uint64_t(group / 4) * scale_rows * 4 + output_row * 4 + group % 4;
          store_scale<ScaleType>(ctx.params.output_scales, scale_index, result.scale);
          if constexpr (kPackedScale && kScaleColumns > kGroups) {
            if (group + 1 == kGroups) {
              PRAGMA_UNROLL
              for (uint32_t pad = 1; pad <= kScaleColumns - kGroups; pad++)
                store_scale<ScaleType>(ctx.params.output_scales, scale_index + pad, ScaleStorage<ScaleType>{});
            }
          }
        }
      } else {
        if constexpr (kUseTmaC) store_values<scalar_t, kValues>(reinterpret_cast<scalar_t *>(destination), values, !valid);
        else if (valid) {
          alignas(16) scalar_t packed[kValues];
          store_values<scalar_t, kValues>(packed, values, false);
          writer.template store_output_packed<kValues * sizeof(scalar_t)>(output_row, output_column, reinterpret_cast<const uint8_t *>(packed));
        }
      }
    }
  }

  template <class WriteChunk>
  CUDA_INLINE void write_umma(MMA &mma, uint32_t slice_id, uint32_t slice_count, WriteChunk write_chunk) {
    // Overlapped accumulators can emit other rows before row zero. Prepare
    // channel scales before either output order reads them.
    arith.may_process_on_smem_write(0, 0);
    // Specialize both output orders so scale-array indices stay compile-time constants.
    if constexpr (MMA::kAccumulatorStride != 0 && MMA::kAccumulatorStride < WarpShape::M) {
      if (mma.output_chunk_index(0) != 0) write_umma_order<true>(mma, slice_id, slice_count, write_chunk);
      else write_umma_order<false>(mma, slice_id, slice_count, write_chunk);
    } else write_umma_order<false>(mma, slice_id, slice_count, write_chunk);
  }

  template <bool kRotate, class WriteChunk>
  CUDA_INLINE void write_umma_order(MMA &mma, uint32_t slice_id, uint32_t slice_count, WriteChunk write_chunk) {
    constexpr bool kChunked = Ctx::kOutputChunkRows != 0;
    constexpr uint32_t kStorageRows = kOutputRows;
    uint32_t lane = ctx.lane_id();
    uint32_t warp = (ctx.math_thread_id() % 128) / 32;
    uint32_t n_partition = ctx.math_group;
    uint32_t column = n_partition * 128 + warp * 32 + (lane / 8) * 8;
    uint32_t row_in_matrix = Ctx::kUmmaCtaGroupSize == 2 ? lane % 8 : (lane % 8) / 2 + (lane % 2) * 4;
    uint32_t smem_base = offsetof(SharedStorage, reduce) / 128 % 8;
    uint32_t output_base = cast_smem_ptr_to_uint(ctx.smem.reduce);
    uint32_t chunk_step = 0;
    constexpr bool kSplitRows = kNumMathThreads == 256;
    uint32_t row_group_offset = kSplitRows ? ctx.math_thread_id() / 128 * 2 : 0;

    // Share fused output code across chunks instead of duplicating activation
    // and requantization expressions for every chunk.
    constexpr uint32_t kChunkUnroll = Ctx::kHasFusedOutput ? 1 : CEIL_DIV(WarpShape::M, 32);
    PRAGMA_UNROLL_COUNT(kChunkUnroll)
    for (uint32_t step = 0; step < CEIL_DIV(WarpShape::M, 32); step++) {
      uint32_t m = kRotate ? (step + MMA::kAccumulatorStride / 32) % CEIL_DIV(WarpShape::M, 32) : step;
      uint32_t lower[16];
      uint32_t upper[16];
      uint32_t tile_rows = MIN(32, WarpShape::M - m * 32);
      uint32_t row_offset = row_group_offset * 8;
      uint32_t rows = row_offset < tile_rows ? MIN(kSplitRows ? 16u : 32u, tile_rows - row_offset) : 0;
      if (rows != 0) mma.load_output_chunk(m, rows, lower, upper, row_offset);
      if constexpr (kChunked && (!Ctx::kIsIndexedGemm || MMA::kCanOverlapAccumulators)) {
        if (step + 1 == MMA::kAccumulatorReleaseChunks && ctx.math_group + 1 == MMA::kOutputGroups) {
          tcgen05_fence_before_thread_sync();
          ctx.sync_math_threads();
          if (ctx.math_thread_id() == 0) {
            if constexpr (Ctx::kUmmaCtaGroupSize == 2)
              mbarrier_arrive<true>(__cluster_map_shared_rank(&ctx.smem.umma_accumulator_free, 0));
            else mbarrier_arrive(&ctx.smem.umma_accumulator_free);
          }
        }
      }
      uint32_t buffer_offset = 0;
      if constexpr (kChunked) {
        buffer_offset = ((chunk_step + output_chunk_phase) % 2) * kStorageRows * BlockShape::N;
        if (m * 32 % kStorageRows == 0) {
          if constexpr (Ctx::kUseTmaC) tma_wait_store_group<1, true>();
          ctx.sync_math_threads();
        }
      }
      PRAGMA_UNROLL
      for (uint32_t group = 0; group < (kSplitRows ? 2 : 4); group++) {
        // Keep fragment indices static even when the final chunk is partial.
        // A runtime loop bound can materialize the TMEM fragments in local memory.
        if (group * 8 >= rows) continue;
        uint32_t first_row = m * 32 + (group + row_group_offset) * 8;
        uint32_t local_row = kChunked ? first_row % kStorageRows : first_row;
        uint32_t values[4];
        // TMEM holds N in rows and M in columns. stmatrix writes four 8-column
        // matrices from the two 16-row TMEM loads.
        if constexpr (Ctx::kUmmaCtaGroupSize == 2) {
          values[0] = convert_umma_pair(lower[group * 4], lower[group * 4 + 1], m * 4 + group + row_group_offset, 0);
          values[1] = convert_umma_pair(lower[group * 4 + 2], lower[group * 4 + 3], m * 4 + group + row_group_offset, 1);
          values[2] = convert_umma_pair(upper[group * 4], upper[group * 4 + 1], m * 4 + group + row_group_offset, 2);
          values[3] = convert_umma_pair(upper[group * 4 + 2], upper[group * 4 + 3], m * 4 + group + row_group_offset, 3);
        } else {
          values[0] = convert_umma_pair(lower[group * 4], lower[group * 4 + 2], m * 4 + group + row_group_offset, 0);
          values[1] = convert_umma_pair(lower[group * 4 + 1], lower[group * 4 + 3], m * 4 + group + row_group_offset, 1);
          values[2] = convert_umma_pair(upper[group * 4], upper[group * 4 + 2], m * 4 + group + row_group_offset, 2);
          values[3] = convert_umma_pair(upper[group * 4 + 1], upper[group * 4 + 3], m * 4 + group + row_group_offset, 3);
        }
        if constexpr (kApplyUmmaActivationInRegisters)
          apply_umma_register_output_arithmetic(values, m * 4 + group + row_group_offset);
        uint32_t row = local_row + row_in_matrix;
        uint32_t swizzled_column = ((column % 64 / 8) ^ ((row + smem_base) % 8)) * 8;
        uint32_t output_offset = buffer_offset + (row + kStorageRows * (column / 64)) * 64 + swizzled_column;
        st_shared<4, true>(output_base + output_offset * 2, values);
      }
      if constexpr (kChunked) {
        uint32_t chunk_end = m * 32 + tile_rows;
        if (chunk_end % kStorageRows == 0 || chunk_end == WarpShape::M) {
          uint32_t chunk_first_row = m * 32 / kStorageRows * kStorageRows;
          uint32_t chunk_rows = chunk_end - chunk_first_row;
          if constexpr (ArithClass::kNeedsPackedOutputTransform && !kApplyUmmaActivationInRegisters) {
            if constexpr (kSplitRows) ctx.sync_math_threads();
            apply_umma_packed_output_arithmetic(chunk_first_row, chunk_rows, buffer_offset / 8, kStorageRows);
          }
          if constexpr (Ctx::kUseTmaC) tma_fence_async_shared();
          ctx.sync_math_threads();
          write_chunk(chunk_first_row, chunk_rows, buffer_offset / 8);
          if constexpr (!Ctx::kUseTmaC) ctx.sync_math_threads();
          chunk_step++;
        }
      }
    }
    // Publish all partial sums before another slice acquires the output lock.
    if constexpr (kChunked && Ctx::kUseTmaC && Ctx::kUseStreamK) {
      if (slice_count > 1 && slice_id != slice_count - 1) tma_wait_store_group<0>();
    }
    if constexpr (kChunked) output_chunk_phase ^= CEIL_DIV(WarpShape::M, kStorageRows) % 2;
    else if constexpr (ArithClass::kNeedsPackedOutputTransform && !kApplyUmmaActivationInRegisters) {
      if constexpr (kSplitRows) ctx.sync_math_threads();
      apply_umma_packed_output_arithmetic();
    }
  }

private:
  CUDA_INLINE void apply_output_activation(uint32_t &value) {
    if constexpr (Ctx::kHasFusedOutput && !Ctx::kRequantizeOutput) {
      using Activation = typename Ctx::LayerConfig::ActivationFunc;
      auto pair = this->num22float2(*reinterpret_cast<scalar_t2 *>(&value));
      if constexpr (kIsBinaryActivation) pair = {Activation::apply(pair.x, pair.y), 0.f};
      else if constexpr (Activation::kType == ActivationType::Unary)
        pair = {Activation::apply(pair.x), Activation::apply(pair.y)};
      auto converted = this->float22num2(pair);
      value = *reinterpret_cast<uint32_t *>(&converted);
    }
  }

  uint32_t output_chunk_phase = 0;

  CUDA_INLINE uint32_t convert_umma_pair(uint32_t first, uint32_t second, uint32_t row_group, uint32_t column_group) {
    float first_value = __uint_as_float(first);
    float second_value = __uint_as_float(second);
    if constexpr (kIsIntAccum) {
      first_value = float(int32_t(first));
      second_value = float(int32_t(second));
    }
    arith.apply_native_f32_output_arithmetic(first_value, second_value, row_group, column_group);
    float2 values = {first_value, second_value};
    auto packed = this->float22num2(values);
    return *reinterpret_cast<uint32_t *>(&packed);
  }

  CUDA_INLINE void apply_umma_register_output_arithmetic(uint32_t (&values)[4], uint32_t row_group) {
    if constexpr (!ArithClass::kNeedsPackedOutputTransform && kIsBinaryActivation) {
      // Adjacent N channels occupy lanes separated by four. Each lane computes
      // one of the pair's M rows so GLU uses the whole warp, then repacks the
      // two results in the gate lane and zeros the interleaved up channel.
      bool is_gate_lane = (ctx.lane_id() & 4) == 0;
      uint32_t shift = is_gate_lane ? 0 : 16;
      PRAGMA_UNROLL
      for (uint32_t pair = 0; pair < 4; pair++) {
        uint32_t value = values[pair];
        uint32_t neighbor = __shfl_xor_sync(0xffffffff, value, 4);
        uint32_t gate = is_gate_lane ? value : neighbor;
        uint32_t up = is_gate_lane ? neighbor : value;
        uint32_t adjacent = ((gate >> shift) & 0xffffu) | ((up >> shift) << 16);
        apply_output_activation(adjacent);
        uint32_t other_row = __shfl_xor_sync(0xffffffff, adjacent, 4);
        values[pair] = is_gate_lane ? (adjacent & 0xffffu) | (other_row << 16) : 0;
      }
      return;
    }
    // Unary activation without packed scales/bias is independent of the TMEM
    // fragment layout. Apply it directly to the already-rounded output values.
    constexpr bool kNeedsRowLayout = ArithClass::kNeedsPackedOutputTransform || kIsBinaryActivation;
    uint32_t lane = ctx.lane_id();
    uint32_t source_row = lane / 4;
    if constexpr (Ctx::kUmmaCtaGroupSize == 1) source_row = (source_row % 4) * 2 + source_row / 4;
    uint32_t source_lane = (lane % 4) * 8 + source_row / 2;
    uint32_t source_shift = (source_row % 2) * 16;
    uint32_t first_row = Ctx::kUmmaCtaGroupSize == 2 ? (lane % 4) * 2 : lane % 4;
    uint32_t second_row = first_row + (Ctx::kUmmaCtaGroupSize == 2 ? 1 : 4);
    uint32_t restore_shift = (lane / 4 % 2) * 16;

    PRAGMA_UNROLL
    for (uint32_t pair = 0; pair < 4; pair++) {
      uint32_t value = values[pair];
      if constexpr (kNeedsRowLayout) {
        // Match the row-wise, adjacent-N pairs formerly read after stmatrix.trans.
        uint32_t first = __shfl_sync(0xffffffff, value, source_lane);
        uint32_t second = __shfl_sync(0xffffffff, value, source_lane + 4);
        value = ((first >> source_shift) & 0xffffu) | ((second >> source_shift) << 16);
      }
      if constexpr (ArithClass::kNeedsPackedOutputTransform)
        arith.may_apply_on_smem_write(value, row_group, pair);
      apply_output_activation(value);
      if constexpr (kNeedsRowLayout) {
        // Restore the fragment distribution expected by the existing store.
        uint32_t first = __shfl_sync(0xffffffff, value, first_row * 4 + lane / 8);
        uint32_t second = __shfl_sync(0xffffffff, value, second_row * 4 + lane / 8);
        value = ((first >> restore_shift) & 0xffffu) | ((second >> restore_shift) << 16);
      }
      values[pair] = value;
    }
  }

  CUDA_INLINE void apply_umma_packed_output_arithmetic(uint32_t first_row = 0, uint32_t rows = WarpShape::M,
      uint32_t buffer_offset = 0, uint32_t storage_rows = BlockShape::M) {
    if constexpr (kNumMathThreads == 256)
      if (ctx.math_thread_id() >= 128) return;
    uint32_t n_partition = ctx.math_group;
    uint32_t warp = ctx.math_thread_id() / 32;
    uint32_t lane = ctx.lane_id();
    uint32_t smem_base = offsetof(SharedStorage, reduce) / 128 % 8;

    __syncwarp();
    for (uint32_t row = lane / 4; row < rows; row += 8) {
      PRAGMA_UNROLL
      for (uint32_t pair = 0; pair < 4; pair++) {
        // Visit stmatrix's transposed groups in the channel loader's register order.
        uint32_t column = n_partition * 128 + warp * 32 + pair * 8;
        uint32_t swizzled_chunk = (column % 64 / 8) ^ ((row + smem_base) % 8);
        uint32_t offset = buffer_offset + (row + storage_rows * (column / 64)) * 8 + swizzled_chunk;
        uint32_t *pairs = reinterpret_cast<uint32_t *>(&ctx.smem.reduce[offset]);
        uint32_t value = pairs[lane % 4];
        arith.may_apply_on_smem_write(value, (first_row + row) / 8, pair);
        pairs[lane % 4] = value;
      }
    }
  }
};

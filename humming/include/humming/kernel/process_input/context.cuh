#pragma once

#include <humming/kernel/process_input/activation.cuh>
#include <humming/kernel/process_input/quantization.cuh>

template <class Config, class TuningConfig, ProcessInputQuantizationPhase Phase, bool kFinalizer = false>
struct ProcessInputContext : Config, TuningConfig {
  using SourceType = typename Config::SourceType;
  using TargetType = typename Config::TargetType;
  using ActivationImpl = typename Config::Activation;
  static constexpr InputQuantizationMode kQuantization = Config::kQuantMode;
  static constexpr bool kUseTilePartition = (!kFinalizer && TuningConfig::kUseTilePartition);
  static constexpr bool kDynamicTokenMode = kQuantization == InputQuantizationMode::DynamicToken;
  static constexpr bool kDynamicGroupMode = kQuantization == InputQuantizationMode::DynamicGroup || kQuantization == InputQuantizationMode::StaticTensorDynamicGroup;
  static constexpr bool kDynamicGroupTokenMode = kQuantization == InputQuantizationMode::DynamicGroupToken;
  static constexpr bool kStaticTensorScale = kQuantization == InputQuantizationMode::StaticTensor || kQuantization == InputQuantizationMode::StaticTensorDynamicGroup;
  static constexpr bool kDynamicGroupScale = kDynamicGroupMode || kDynamicGroupTokenMode;
  static constexpr bool kStagedGroupToken = kDynamicGroupTokenMode && TuningConfig::kUseTilePartition;

  static constexpr bool kDynamicScale = kDynamicTokenMode || kDynamicGroupScale;
  static constexpr bool kFusedGroupToken = kDynamicGroupTokenMode && !kStagedGroupToken;
  using ConfiguredDynamicGroupScaleType = std::conditional_t<kDynamicGroupScale, typename Config::GroupScaleType, Float32>;
  using DynamicGroupScaleType = std::conditional_t<kStagedGroupToken, M3BFloat16, ConfiguredDynamicGroupScaleType>;
  using OutputScaleType = std::conditional_t<kDynamicGroupScale, DynamicGroupScaleType, Float32>;
  using QuantScaleType = std::conditional_t<kFusedGroupToken, M3BFloat16, OutputScaleType>;

  static constexpr uint32_t kHiddenSize = Config::kHiddenSize;
  static constexpr uint32_t kQuantGroupSize = Config::kQuantGroupSize;
  static constexpr uint32_t kHadamardBlockSize = Config::kHadamardBlockSize;
  static constexpr bool kHadamard = kHadamardBlockSize > 1;
  static constexpr uint32_t kThreadsPerTask = (kFinalizer ? 32 : TuningConfig::kThreadsPerTask);
  static constexpr uint32_t kValuesPerThread = (kFinalizer ? 1 : TuningConfig::kValuesPerThread);
  static constexpr uint32_t kTokensPerBlock = (kFinalizer ? TuningConfig::kFinalizeTokensPerBlock : TuningConfig::kTokensPerBlock);
  static constexpr uint32_t kThreads = kThreadsPerTask * kTokensPerBlock;
  static constexpr ProcessInputLayoutType kLayout = Config::kLayout;
  static constexpr bool kScatterSingleOutput = TuningConfig::kSeparateOutputs;
  static constexpr ActivationType kActivation = ActivationImpl::kType;
  static constexpr bool kBinaryActivation = kActivation == ActivationType::BinarySplit || kActivation == ActivationType::BinaryInterleaved;
  static constexpr uint32_t kInputRowSize = kHiddenSize * (kBinaryActivation ? 2 : 1);
  static constexpr ProcessInputQuantizationPhase kPhase = Phase;
  static constexpr bool kUseMMajorInputScale = Config::kUseMMajorInputScale;
  static constexpr bool kUsePdl = TuningConfig::kUsePdl;
  static constexpr bool kQuantize = kQuantization != InputQuantizationMode::Disabled;
  static constexpr uint32_t kOutputPacking = kQuantize ? 8 / TargetType::kBits : 1;
  static constexpr uint32_t kColumnsPerTask = kThreadsPerTask * kValuesPerThread;
  static constexpr uint32_t kBlocksPerRow = kUseTilePartition ? (kHiddenSize + kColumnsPerTask - 1) / kColumnsPerTask : 1;
  static constexpr uint32_t kOutputsPerToken = kLayout == ProcessInputLayoutType::Scatter && !kScatterSingleOutput ? Config::kScatterWidth : 1;
  static constexpr bool kFullColumns = kHiddenSize % kColumnsPerTask == 0;
  static constexpr uint32_t kNumWarps = kThreads / 32;
  static constexpr uint32_t kScaleSize = kDynamicTokenMode ? kColumnsPerTask : kQuantGroupSize;
  static constexpr bool kReduce = kQuantize && kDynamicScale && kPhase != ProcessInputQuantizationPhase::Quantize;
  static constexpr uint32_t kTransformScratch = !kFinalizer && kHadamard && kHadamardBlockSize / kValuesPerThread > 32 ? kThreads * kValuesPerThread : 0;
  static constexpr uint32_t kGroupReduceScratch = kReduce && kScaleSize / kValuesPerThread > 32 ? kNumWarps : 0;
  static constexpr uint32_t kTokenReduceScratch = kFusedGroupToken && kThreadsPerTask > 32 ? kNumWarps : 0;
  static constexpr uint32_t kReduceScratch = kGroupReduceScratch > kTokenReduceScratch ? kGroupReduceScratch : kTokenReduceScratch;
  static constexpr uint32_t kScratchElements = kTransformScratch + kReduceScratch > 0 ? kTransformScratch + kReduceScratch : 1;

  struct SharedStorage { float scratch[kFinalizer ? 1 : kScratchElements]; };

  static_assert(supported_scale_type<DynamicGroupScaleType>);
  static_assert(kDynamicGroupScale || std::is_same<DynamicGroupScaleType, Float32>::value);
  static_assert(kPhase == ProcessInputQuantizationPhase::Fused || kDynamicTokenMode);
  static_assert(!std::is_same<DynamicGroupScaleType, M3BFloat16>::value || kPhase == ProcessInputQuantizationPhase::Fused);
  static_assert(!kUseMMajorInputScale || kDynamicScale);
  static_assert(kQuantize || !kDynamicScale);
  static_assert(kQuantize || !kStaticTensorScale);
  static_assert(kQuantize || kPhase == ProcessInputQuantizationPhase::Fused);
  static_assert(kQuantize || !kUseMMajorInputScale);
  static_assert(!kUseTilePartition || (!kDynamicTokenMode && !kFusedGroupToken));
  static_assert(!kFusedGroupToken || kPhase == ProcessInputQuantizationPhase::Fused);
  static_assert(!kUseTilePartition || kTokensPerBlock == 1);
  static_assert(!kScatterSingleOutput || kLayout == ProcessInputLayoutType::Scatter);

  using Activation = InputActivation<ActivationImpl, kHiddenSize, kValuesPerThread>;

  static_assert(kThreads >= 32 && kThreads <= 1024 && kThreads % 32 == 0);
  static_assert(kValuesPerThread >= 1 && (kValuesPerThread & (kValuesPerThread - 1)) == 0);
  static_assert(kFinalizer || kHiddenSize % kValuesPerThread == 0);
  static_assert(kFinalizer || kUseTilePartition || kColumnsPerTask >= kHiddenSize);
  static_assert(kFinalizer || !kUseTilePartition || !kDynamicGroupScale || kColumnsPerTask % kQuantGroupSize == 0);
  static_assert(kFinalizer || !kHadamard || (kHadamardBlockSize <= 512 && kHadamardBlockSize >= 2 &&
      (kHadamardBlockSize & (kHadamardBlockSize - 1)) == 0 && kHiddenSize % kHadamardBlockSize == 0 &&
      kHadamardBlockSize % kValuesPerThread == 0 && (!kUseTilePartition || kColumnsPerTask % kHadamardBlockSize == 0)));
  static_assert(!kDynamicGroupScale || (kQuantGroupSize >= 2 && kQuantGroupSize <= 512 &&
      (kQuantGroupSize & (kQuantGroupSize - 1)) == 0));
  static_assert(Config::kScatterWidth > 0);
  static_assert(kLayout == ProcessInputLayoutType::Scatter || Config::kScatterWidth == 1);

  SharedStorage &smem;
  uint64_t input_row;
  uint32_t column;
  uint64_t output_rows[kOutputsPerToken];
  bool zero_outputs[kOutputsPerToken];
  bool load;
  bool zero;

  CUDA_INLINE ProcessInputContext(
      SharedStorage &shared,
      const void *expert_tokens,
      const void *scatter_idx,
      const void *num_valid_tokens,
      uint64_t num_input_rows,
      uint64_t num_output_rows,
      uint32_t max_tokens_per_expert,
      bool use_int64_expert_tokens,
      bool use_int64_scatter_idx,
      bool use_int64_num_valid_tokens)
      : smem(shared), load(false), zero(false) {
    uint64_t logical_row = (static_cast<uint64_t>(blockIdx.x) / kBlocksPerRow) * kTokensPerBlock + threadIdx.x / kThreadsPerTask;
    column = kFinalizer ? 0 : (blockIdx.x % kBlocksPerRow) * kColumnsPerTask + (threadIdx.x % kThreadsPerTask) * kValuesPerThread;
    input_row = kScatterSingleOutput ? logical_row / Config::kScatterWidth : logical_row;
    PRAGMA_UNROLL
    for (uint32_t route = 0; route < kOutputsPerToken; ++route) {
      output_rows[route] = ~uint64_t{0};
      zero_outputs[route] = false;
    }
    if (input_row >= num_input_rows) return;
    int64_t valid_tokens = static_cast<int64_t>(num_input_rows * Config::kScatterWidth);
    if (num_valid_tokens != nullptr) {
      if (use_int64_num_valid_tokens) valid_tokens = *reinterpret_cast<const int64_t *>(num_valid_tokens);
      else valid_tokens = *reinterpret_cast<const int32_t *>(num_valid_tokens);
    }
    if constexpr (kLayout == ProcessInputLayoutType::Scatter) {
      uint64_t scatter_size = num_input_rows * Config::kScatterWidth;
      uint32_t first_route = kScatterSingleOutput ? logical_row % Config::kScatterWidth : 0;
      PRAGMA_UNROLL
      for (uint32_t route = 0; route < kOutputsPerToken; ++route) {
        uint64_t index = input_row * Config::kScatterWidth + first_route + route;
        int64_t output_row;
        if (use_int64_scatter_idx) output_row = reinterpret_cast<const int64_t *>(scatter_idx)[index];
        else output_row = reinterpret_cast<const int32_t *>(scatter_idx)[index];
        uint64_t row = static_cast<uint64_t>(output_row);
        if (row >= scatter_size) continue;
        if (output_row < valid_tokens) {
          output_rows[route] = row;
          load = true;
        } else if constexpr (Config::kZeroInvalid) {
          output_rows[route] = row;
          zero_outputs[route] = true;
          zero = true;
        }
      }
    } else if (input_row < num_output_rows) {
      load = static_cast<int64_t>(input_row) < valid_tokens;
      if constexpr (kLayout == ProcessInputLayoutType::GroupedMask) {
        uint32_t expert = input_row / max_tokens_per_expert;
        uint32_t local_row = input_row % max_tokens_per_expert;
        int64_t valid_rows;
        if (use_int64_expert_tokens) valid_rows = reinterpret_cast<const int64_t *>(expert_tokens)[expert];
        else valid_rows = reinterpret_cast<const int32_t *>(expert_tokens)[expert];
        load = static_cast<int64_t>(local_row) < valid_rows;
      }
      zero = !load && Config::kZeroInvalid;
      zero_outputs[0] = zero;
      if (load || zero) output_rows[0] = input_row;
    }
  }
};

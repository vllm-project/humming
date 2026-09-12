#pragma once

#include <humming/kernel/process_input/activation.cuh>
#include <humming/kernel/process_input/hadamard.cuh>
#include <humming/kernel/process_input/context.cuh>
#include <humming/kernel/process_input/thread_task.cuh>
#include <humming/kernel/process_input/quantization.cuh>




template <class Config, class ScaleType>
CUDA_INLINE uint64_t group_scale_index(uint64_t output_row, uint32_t group, uint64_t scale_stride) {
  constexpr uint64_t kGroupsPerToken = Config::kHiddenSize / Config::kQuantGroupSize;
  if constexpr (std::is_same<ScaleType, M3BFloat16>::value) {
    return output_row * kGroupsPerToken + group;
  } else if constexpr (!Config::kUseMMajorInputScale) {
    constexpr uint32_t kStride = std::is_same<ScaleType, Float32>::value ? kGroupsPerToken : (kGroupsPerToken + 3) / 4 * 4;
    return output_row * kStride + group;
  } else if constexpr (std::is_same<ScaleType, Float32>::value) {
    return group * scale_stride + output_row;
  } else {
    static_assert(std::is_same<ScaleType, Float8E4M3>::value || std::is_same<ScaleType, Float8E8M0>::value);
    return (group / 4) * (scale_stride * 4) + output_row * 4 + group % 4;
  }
}


template <class BaseConfig, class TuningConfig, ProcessInputQuantizationPhase Phase>
__global__ __launch_bounds__(TuningConfig::kThreadsPerTask * TuningConfig::kTokensPerBlock) void process_input_kernel(
    const void *input,
    void *output,
    const float *static_tensor_scales,
    void *output_scales,
    float *token_scales,
    const void *expert_tokens,
    const void *scatter_idx,
    const void *num_valid_tokens,
    uint64_t num_input_rows,
    uint64_t num_output_rows,
    uint32_t max_tokens_per_expert,
    uint64_t group_scale_stride,
    bool use_int64_expert_tokens,
    bool use_int64_scatter_idx,
    bool use_int64_num_valid_tokens) {
  using Config = ProcessInputContext<BaseConfig, TuningConfig, Phase>;
  using SourceType = typename Config::SourceType;
  using TargetType = typename Config::TargetType;
  using Activation = typename Config::Activation;
  using OutputScaleType = typename Config::OutputScaleType;
  using QuantScaleType = typename Config::QuantScaleType;

  constexpr uint32_t K = Config::kHiddenSize;
  constexpr uint32_t G = Config::kQuantGroupSize;
  constexpr uint32_t H = Config::kHadamardBlockSize;
  constexpr uint32_t V = Config::kValuesPerThread;
  constexpr uint32_t kThreads = Config::kThreads;
  constexpr uint32_t kNumWarps = kThreads / 32;
  constexpr uint32_t kPackedBytes = V * TargetType::kBits / 8;
  constexpr uint32_t kTransformScratch = Config::kTransformScratch;
  constexpr uint32_t kReduceScratch = Config::kReduceScratch;
  constexpr bool kRawCopy = !Config::kQuantize && Config::kActivation == ActivationType::None && !Config::kHadamard;
  constexpr bool kPureHadamard =
      !Config::kQuantize &&
      Config::kActivation == ActivationType::None &&
      Config::kHadamard &&
      Config::kLayout == ProcessInputLayoutType::Normal &&
      Config::kUseTilePartition &&
      Config::kFullColumns;

  static_assert(kThreads >= 32 && kThreads <= 1024 && kThreads % 32 == 0);
  static_assert(Config::kUseTilePartition || Config::kThreadsPerTask * V >= K);
  static_assert(K % G == 0 && G % V == 0);
  static_assert(V >= 1 && (V & (V - 1)) == 0);
  static_assert(!Config::kQuantize || V * TargetType::kBits % 8 == 0);
  if constexpr (Config::kHadamard) {
    static_assert(H >= 2 && (H & (H - 1)) == 0);
    static_assert(K % H == 0 && H % V == 0);
    static_assert(!Config::kUseTilePartition || Config::kColumnsPerTask % H == 0);
  }

  if constexpr (Config::kUsePdl) griddepcontrol_wait();

  __shared__ typename Config::SharedStorage shared;
  Config ctx(
      shared, expert_tokens, scatter_idx, num_valid_tokens, num_input_rows, num_output_rows,
      max_tokens_per_expert, use_int64_expert_tokens, use_int64_scatter_idx, use_int64_num_valid_tokens);
  float *scratch = shared.scratch;
  auto thread = ProcessInputThreadTask<Config>(ctx);

  if constexpr (kRawCopy) {
    alignas(16) SourceType values[V];
    if (!thread.skip_read) {
      const SourceType *input2 = reinterpret_cast<const SourceType *>(input);
      load_raw_values<SourceType, V>(values, thread.input(input2));
    }

    PRAGMA_UNROLL
    for (uint32_t route = 0; route < Config::kOutputsPerToken; route++) {
      auto write = ProcessInputThreadTask<Config>(ctx, route);
      if (!write.skip_output) {
        SourceType *destination = reinterpret_cast<SourceType *>(write.template output<sizeof(SourceType) * 8>(output));
        store_raw_values<SourceType, V>(destination, values, write.zero_output());
      }
    }
  } else {
    float values[V];
    if (!thread.skip_read) {
      const SourceType *input2 = reinterpret_cast<const SourceType *>(input);
      if constexpr (Config::kActivation == ActivationType::None) {
        load_values<SourceType, V>(values, thread.input(input2));
      } else {
        Activation::template load<SourceType>(values, input2, thread.input_row, thread.input_col);
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < V; value++)
        values[value] = 0.f;
    }

    if constexpr (Config::kHadamard) {
      constexpr bool kTileBarriers = !kPureHadamard && H / V >= 128;
      hadamard<V, H, kNumWarps, 0, kTileBarriers>(values, scratch);
      if constexpr (kTransformScratch > 0 && kReduceScratch > 0) __syncthreads();
    }

    if constexpr (Config::kQuantize) {
      ScaleStorage<QuantScaleType> collected_scale{};
      if constexpr (Config::kPhase == ProcessInputQuantizationPhase::Quantize) {
        PRAGMA_UNROLL
        for (uint32_t route = 0; route < Config::kOutputsPerToken; route++) {
          auto write = ProcessInputThreadTask<Config>(ctx, route);
          if (!write.skip_output && !write.zero_output()) {
            collected_scale = load_scale<Float32>(output_scales, write.output_row);
            break;
          }
        }
      }

      uint32_t group = thread.input_col < K ? thread.input_col / G : 0;
      float static_scale = 1.f;
      if constexpr (Config::kStaticTensorScale)
        static_scale *= __ldg(static_tensor_scales);

      auto result = quant_group<Config>(values, scratch + kTransformScratch, static_scale, collected_scale);
      ScaleStorage<OutputScaleType> group_output_scale{};
      float token_scale = 0.f;
      if constexpr (Config::kFusedGroupToken) {
        float m3_scale = decode_scale<QuantScaleType>(result.scale);
        float token_maximum = token_group_scale_max<
            G,
            V,
            Config::kThreadsPerTask,
            kNumWarps,
            0>(m3_scale, scratch + kTransformScratch);
        token_scale = token_scale_from_m3_bits(__float_as_uint(token_maximum));
        float local_scale = token_scale > 0.f ? m3_scale / token_scale : 0.f;
        group_output_scale = encode_scale<OutputScaleType>(local_scale);
      } else {
        group_output_scale = result.scale;
      }
      bool scale_leader;
      if constexpr (Config::kDynamicTokenMode) {
        scale_leader = thread.input_col == 0;
      } else {
        scale_leader = thread.input_col % G == 0;
      }

      PRAGMA_UNROLL
      for (uint32_t route = 0; route < Config::kOutputsPerToken; route++) {
        auto write = ProcessInputThreadTask<Config>(ctx, route);
        if (!write.skip_output) {
          if constexpr (Config::kReduce) {
            if (scale_leader) {
              uint64_t scale_index = write.output_row;
              if constexpr (Config::kDynamicGroupScale) {
                scale_index = group_scale_index<Config, OutputScaleType>(write.output_row, group, group_scale_stride);
              }
              ScaleStorage<OutputScaleType> stored_scale = write.zero_output() ? ScaleStorage<OutputScaleType>{} : group_output_scale;
              store_scale<OutputScaleType>(output_scales, scale_index, stored_scale);
            }
            if constexpr (Config::kFusedGroupToken)
              if (thread.input_col == 0) token_scales[write.output_row] = write.zero_output() ? 0.f : token_scale;
          }
          if constexpr (Config::kPhase != ProcessInputQuantizationPhase::CollectAbsmax) {
            store_packed<kPackedBytes>(write.template output<TargetType::kBits>(output), result.packed, write.zero_output());
          }
        }
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t route = 0; route < Config::kOutputsPerToken; route++) {
        auto write = ProcessInputThreadTask<Config>(ctx, route);
        if (!write.skip_output) {
          SourceType *destination = reinterpret_cast<SourceType *>(write.template output<sizeof(SourceType) * 8>(output));
          store_values<SourceType, V>(destination, values, write.zero_output());
        }
      }
    }
  }

  if constexpr (Config::kUsePdl) {
    __syncthreads();
    if (threadIdx.x == 0) griddepcontrol_launch_dependents();
  }
}


// Scale-only second stage for DynamicGroup E4M3 + DynamicToken.  One warp
// owns one logical token; layout routing is repeated without reading input.
template <class BaseConfig, class TuningConfig, uint32_t kTokensPerBlock>
__global__ __launch_bounds__(kTokensPerBlock * 32) void finalize_group_token_scales_kernel(
    const uint16_t *input,
    void *output_scales,
    float *token_scales,
    const void *expert_tokens,
    const void *scatter_idx,
    const void *num_valid_tokens,
    uint64_t num_input_rows,
    uint64_t num_output_rows,
    uint32_t max_tokens_per_expert,
    uint64_t group_scale_stride,
    bool use_int64_expert_tokens,
    bool use_int64_scatter_idx,
    bool use_int64_num_valid_tokens) {
  using Config = ProcessInputContext<BaseConfig, TuningConfig, ProcessInputQuantizationPhase::Fused, true>;
  using OutputScaleType = typename Config::ConfiguredDynamicGroupScaleType;
  constexpr uint32_t kGroupsPerToken = Config::kHiddenSize / Config::kQuantGroupSize;
  static_assert(Config::kStagedGroupToken);
  static_assert(kTokensPerBlock == Config::kTokensPerBlock && kTokensPerBlock <= 32);
  if constexpr (Config::kUsePdl) griddepcontrol_wait();
  __shared__ typename Config::SharedStorage shared;
  Config ctx(
      shared, expert_tokens, scatter_idx, num_valid_tokens, num_input_rows, num_output_rows,
      max_tokens_per_expert, use_int64_expert_tokens, use_int64_scatter_idx, use_int64_num_valid_tokens);
  uint32_t lane = threadIdx.x & 31;
  uint64_t source_row = ~uint64_t{0};
  PRAGMA_UNROLL
  for (uint32_t route = 0; route < Config::kOutputsPerToken; ++route) {
    auto task = ProcessInputThreadTask<Config>(ctx, route);
    if (!task.skip_output && !task.zero_output() && source_row == ~uint64_t{0}) source_row = task.output_row;
  }
  bool active = source_row != ~uint64_t{0};
  uint32_t maximum = 0;
  if (active)
    for (uint32_t group = lane; group < kGroupsPerToken; group += 32)
      maximum = max(maximum, static_cast<uint32_t>(input[source_row * kGroupsPerToken + group]));
#if __CUDA_ARCH__ >= 800
  maximum = warp_max(maximum);
#else
  PRAGMA_UNROLL
  for (uint32_t step = 16; step >= 1; step >>= 1)
    maximum = max(maximum, __shfl_down_sync(0xFFFFFFFFu, maximum, step));
  maximum = __shfl_sync(0xFFFFFFFFu, maximum, 0);
#endif
  float token_scale = maximum == 0 ? 0.f : token_scale_from_m3(static_cast<uint16_t>(maximum));
  PRAGMA_UNROLL
  for (uint32_t route = 0; route < Config::kOutputsPerToken; ++route) {
    auto task = ProcessInputThreadTask<Config>(ctx, route);
    if (!task.skip_output) {
      if (lane == 0) token_scales[task.output_row] = task.zero_output() ? 0.f : token_scale;
      for (uint32_t group = lane; group < kGroupsPerToken; group += 32) {
        float scale = !task.zero_output() && token_scale > 0.f ? decode_scale<M3BFloat16>(input[source_row * kGroupsPerToken + group]) / token_scale : 0.f;
        uint64_t index = group_scale_index<Config, OutputScaleType>(task.output_row, group, group_scale_stride);
        store_scale<OutputScaleType>(output_scales, index, encode_scale<OutputScaleType>(scale));
      }
    }
  }
  if constexpr (Config::kUsePdl) {
    __syncthreads();
    if (threadIdx.x == 0) griddepcontrol_launch_dependents();
  }
}

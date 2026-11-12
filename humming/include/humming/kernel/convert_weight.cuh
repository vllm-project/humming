#pragma once

#include <humming/datatype/dtypes.cuh>


template <class SourceType, class TargetType>
CUDA_INLINE uint32_t convert_weight_code(uint32_t code, bool &can_convert) {
  constexpr int64_t midpoint = int64_t{1} << (SourceType::kBits - 1);
  constexpr uint32_t mask = static_cast<uint32_t>(2 * midpoint - 1);
  code &= mask;
  int64_t value;
  if constexpr (SourceType::kIsIntegerType) {
    if constexpr (SourceType::kIsSigned) {
      value = (code & midpoint) ? int64_t{code} - 2 * midpoint : int64_t{code};
    } else {
      value = int64_t{code} - midpoint;
    }
  } else {
    int64_t magnitude = code & (midpoint - 1);
    value = (code & midpoint) ? -magnitude : magnitude;
  }

  if constexpr (TargetType::kIsIntegerType) {
    if constexpr (!TargetType::kIsSigned) value += midpoint;
    return static_cast<uint32_t>(value);
  } else {
    int64_t magnitude = value < 0 ? -value : value;
    if (magnitude >= midpoint) {
      can_convert = false;
      magnitude = midpoint - 1;
    }
    uint32_t sign = value < 0 && magnitude != 0 ? static_cast<uint32_t>(midpoint) : 0;
    return sign | static_cast<uint32_t>(magnitude);
  }
}


template <class SourceType, class TargetType, bool kCheckOnly = false>
__global__ void convert_weight_kernel(const uint32_t *inputs, uint32_t *outputs, int64_t num_elements) {
  static_assert(SourceType::kBits == TargetType::kBits);
  static_assert(SourceType::kBits >= 1 && SourceType::kBits <= 32);
  if constexpr (SourceType::kIsFloatingPointType) {
    static_assert(SourceType::kIsSigned && SourceType::kExponentBits == 0);
  }
  if constexpr (TargetType::kIsFloatingPointType) {
    static_assert(TargetType::kIsSigned && TargetType::kExponentBits == 0);
  }

  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool can_convert = true;
  constexpr uint32_t kBits = SourceType::kBits;
  constexpr uint32_t kDivisor = kBits & -kBits; // gcd(kBits, 32) for 1 <= kBits <= 32.
  constexpr uint32_t kWords = kBits / kDivisor;
  constexpr uint32_t kValues = 32 / kDivisor;
  constexpr uint32_t kMask = static_cast<uint32_t>((uint64_t{1} << kBits) - 1);
  int64_t offset = index * kWords;
  if (offset >= num_elements) return;

  // Each thread owns complete words, including codes crossing word boundaries.
  uint32_t input_words[kWords];
  uint32_t output_words[kWords] = {};
  for (uint32_t i = 0; i < kWords; ++i)
    input_words[i] = inputs[offset + i];
  for (uint32_t i = 0; i < kValues; ++i) {
    uint32_t word_index = i * kBits / 32;
    uint32_t bit_offset = i * kBits % 32;
    uint32_t code = input_words[word_index] >> bit_offset;
    if (bit_offset + kBits > 32) code |= input_words[word_index + 1] << (32 - bit_offset);
    uint32_t converted = convert_weight_code<SourceType, TargetType>(code, can_convert) & kMask;
    if constexpr (!kCheckOnly) {
      output_words[word_index] |= converted << bit_offset;
      if (bit_offset + kBits > 32) output_words[word_index + 1] |= converted >> (32 - bit_offset);
    }
  }
  if constexpr (!kCheckOnly) {
    for (uint32_t i = 0; i < kWords; ++i)
      outputs[offset + i] = output_words[i];
  }
  if constexpr (kCheckOnly) {
    if (!can_convert) atomicExch(outputs, 0u);
  }
}

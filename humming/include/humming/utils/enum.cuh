#pragma once


enum class InputQuantizationMode : uint32_t {
  Disabled = 0,
  StaticTensor = 1,
  DynamicToken = 2,
  DynamicGroup = 3,
  StaticTensorDynamicGroup = 4,
  DynamicGroupToken = 5,
};


enum class WeightScaleType : uint32_t {
  GROUP,
  BLOCK,
  CHANNEL,
  TENSOR,
};


enum class WeightScale2Type : uint32_t {
  NONE,
  CHANNEL,
  TENSOR,
};


enum class MmaType : uint32_t {
  MMA,
  WGMMA,
  MXMMA
};


enum class GemmType : uint32_t {
  DENSE,
  INDEXED,
  GROUPED_CONTIGUOUS,
  GROUPED_MASKED,
};

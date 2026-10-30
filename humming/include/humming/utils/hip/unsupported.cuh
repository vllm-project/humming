#pragma once
// Error-on-use stubs for NVIDIA-only primitives (TMA, WGMMA, mbarrier, ldmatrix, cp.async
// bulk/AIU, clusters) on ROCm.
//
// A stub keeps the generic headers parseable by hipcc. Any kernel that actually emits a call
// to one fails to compile with "NVIDIA-only primitive used on ROCm" instead of silently
// miscompiling; unused stubs cost nothing.

#include <humming/utils/platform.cuh>

__device__ __attribute__((error("NVIDIA-only primitive used on ROCm"))) void humming_rocm_unsupported_primitive();

// Defines `name` as a variadic template that accepts any explicit template arguments and any
// call arguments, so call sites written for the CUDA signature still parse.
#define HUMMING_ROCM_UNSUPPORTED_STUB(ReturnType, name)          \
  template <auto... kTemplateArgs, typename... Args>             \
  __device__ __forceinline__ ReturnType name(Args &&...) {       \
    humming_rocm_unsupported_primitive();                        \
    return ReturnType();                                         \
  }

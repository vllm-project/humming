#pragma once
// Platform layer: the only header that knows whether Humming is compiled for CUDA or ROCm.
//
// CUDA: includes <cuda.h> and nothing else, so NVIDIA builds are unchanged.
// ROCm: maps the CUDA headers/types/intrinsics used by the portable kernels onto HIP.
//
// Contract for device code:
//   HUMMING_HIP            defined (to 1) when compiling for AMD GPUs.
//   HUMMING_NAMED_BARRIER_SYNC(id, threads)
//                          bar.sync on CUDA. ROCm has no named barriers: falls back to
//                          __syncthreads(), which is only correct when every thread of the block
//                          executes the same sequence of named-barrier syncs.
// ROCm only (the CUDA side defines nothing else, so NVIDIA PTX is unchanged):
//   kWarpSize              hardware wavefront width (64 on CDNA).
//   kLogicalWarpSize       32. Portable kernels are written for 32-lane warps; on ROCm a
//                          64-lane wavefront holds two logical warps (lanes 0-31, 32-63).
//   kFullWarpMask          all-lanes mask of the hardware wavefront (uint64_t).
//   __shfl_*_sync, __reduce_{add,max,min}_sync, __ballot_sync, __syncwarp
//                          On ROCm these act on the 32-lane logical warp (default width 32);
//                          the mask is accepted in any integer width and ignored: callers
//                          must pass the full logical warp (all 32 lanes active and converged).
//   HUMMING_DEVICE_ASSERT_FAIL(message)   device-side trap.

#include <cstdint>

#if !defined(HUMMING_HIP) && (defined(__HIP__) || defined(__HIPCC__))
#define HUMMING_HIP 1
#endif

#ifndef HUMMING_HIP

#include <cuda.h>

#define HUMMING_NAMED_BARRIER_SYNC(barrier_id, num_threads) \
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads) : "memory")

#else  // HUMMING_HIP

// Humming provides the CUDA *_sync warp functions itself (32-lane logical warps, any mask
// width); HIP's versions require 64-bit masks and span the whole 64-lane wavefront.
#ifndef HIP_DISABLE_WARP_SYNC_BUILTINS
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

#include <cstdint>
#include <type_traits>

constexpr uint32_t kWarpSize = 64;
constexpr uint32_t kLogicalWarpSize = 32;
constexpr uint64_t kFullWarpMask = ~uint64_t{0};

// ---------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
// gfx950 implements the OCP FP8 formats, which match the NVIDIA e4m3/e5m2 encodings.
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;

// <cuda.h> host type referenced by kernel signatures. TMA itself is never used on ROCm.
#ifndef CU_TENSOR_MAP_NUM_QWORDS
#define CU_TENSOR_MAP_NUM_QWORDS 16
typedef uint64_t cuuint64_t;
typedef struct CUtensorMap_st {
  alignas(64) cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
} CUtensorMap;
#endif

// ---------------------------------------------------------------------------------------
// Conversions missing from hip_bf16.h
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ __hip_bfloat162 __float2bfloat162_rn(float value) {
  __hip_bfloat16 converted = __float2bfloat16(value);
  return __halves2bfloat162(converted, converted);
}

__device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float low, float high) {
  return __halves2bfloat162(__float2bfloat16(low), __float2bfloat16(high));
}

// ---------------------------------------------------------------------------------------
// Integer / memory intrinsics
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ int __dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c) {
  return __builtin_amdgcn_udot4(a, b, c, false);
}

// LDS byte address of a generic pointer that points into shared memory.
template <typename T>
__device__ __forceinline__ size_t __cvta_generic_to_shared(T *pointer) {
  using SharedPointer = __attribute__((address_space(3))) const void *;
  auto shared_pointer = (SharedPointer)(const_cast<const void *>(reinterpret_cast<volatile const void *>(pointer)));
  return static_cast<size_t>(reinterpret_cast<uintptr_t>(shared_pointer));
}

// Streaming / L2-only cache hints are performance hints; plain accesses are equivalent.
template <typename T>
__device__ __forceinline__ T __ldcs(const T *pointer) {
  return *pointer;
}

template <typename T>
__device__ __forceinline__ T __ldcg(const T *pointer) {
  return *pointer;
}

template <typename T>
__device__ __forceinline__ void __stcg(T *pointer, T value) {
  *pointer = value;
}

template <typename T>
__device__ __forceinline__ void __stcs(T *pointer, T value) {
  *pointer = value;
}

// ---------------------------------------------------------------------------------------
// 32-lane logical warp collectives
// ---------------------------------------------------------------------------------------
namespace humming_platform {

template <typename T>
__device__ __forceinline__ T shuffle_index(T value, int source_lane) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
  if constexpr (sizeof(T) == 8) {
    uint64_t bits = __builtin_bit_cast(uint64_t, value);
    uint32_t low = __builtin_amdgcn_ds_bpermute(source_lane << 2, static_cast<int>(bits));
    uint32_t high = __builtin_amdgcn_ds_bpermute(source_lane << 2, static_cast<int>(bits >> 32));
    return __builtin_bit_cast(T, (static_cast<uint64_t>(high) << 32) | low);
  } else if constexpr (sizeof(T) == 4) {
    int bits = __builtin_bit_cast(int, value);
    return __builtin_bit_cast(T, __builtin_amdgcn_ds_bpermute(source_lane << 2, bits));
  } else if constexpr (sizeof(T) == 2) {
    int bits = __builtin_bit_cast(uint16_t, value);
    return __builtin_bit_cast(T, static_cast<uint16_t>(__builtin_amdgcn_ds_bpermute(source_lane << 2, bits)));
  } else {
    int bits = __builtin_bit_cast(uint8_t, value);
    return __builtin_bit_cast(T, static_cast<uint8_t>(__builtin_amdgcn_ds_bpermute(source_lane << 2, bits)));
  }
}

__device__ __forceinline__ int hardware_lane() { return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0)); }

}  // namespace humming_platform

// CUDA semantics with width <= 32: lanes are grouped in segments of `width` inside each
// 32-lane logical warp. Out-of-segment sources return the caller's own value for up/down.
template <typename MaskT, typename T>
__device__ __forceinline__ T __shfl_sync(MaskT, T value, int source_lane, int width = kLogicalWarpSize) {
  int lane = humming_platform::hardware_lane();
  int segment_base = lane & ~(width - 1);
  int target = segment_base + (source_lane & (width - 1));
  return humming_platform::shuffle_index(value, target);
}

template <typename MaskT, typename T>
__device__ __forceinline__ T __shfl_xor_sync(MaskT, T value, int lane_mask, int width = kLogicalWarpSize) {
  int lane = humming_platform::hardware_lane();
  int segment_base = lane & ~(width - 1);
  int xor_lane = lane ^ lane_mask;
  bool inside_segment = xor_lane >= segment_base && xor_lane < segment_base + width;
  int target = inside_segment ? xor_lane : lane;
  return humming_platform::shuffle_index(value, target);
}

template <typename MaskT, typename T>
__device__ __forceinline__ T __shfl_down_sync(MaskT, T value, unsigned int delta, int width = kLogicalWarpSize) {
  int lane = humming_platform::hardware_lane();
  int segment_lane = lane & (width - 1);
  int target = segment_lane + static_cast<int>(delta) < width ? lane + static_cast<int>(delta) : lane;
  return humming_platform::shuffle_index(value, target);
}

template <typename MaskT, typename T>
__device__ __forceinline__ T __shfl_up_sync(MaskT, T value, unsigned int delta, int width = kLogicalWarpSize) {
  int lane = humming_platform::hardware_lane();
  int segment_lane = lane & (width - 1);
  int target = segment_lane >= static_cast<int>(delta) ? lane - static_cast<int>(delta) : lane;
  return humming_platform::shuffle_index(value, target);
}

template <typename MaskT, typename T>
__device__ __forceinline__ T __reduce_max_sync(MaskT mask, T value) {
  for (int step = 1; step < static_cast<int>(kLogicalWarpSize); step <<= 1) {
    T other = __shfl_xor_sync(mask, value, step);
    value = other > value ? other : value;
  }
  return value;
}

template <typename MaskT, typename T>
__device__ __forceinline__ T __reduce_min_sync(MaskT mask, T value) {
  for (int step = 1; step < static_cast<int>(kLogicalWarpSize); step <<= 1) {
    T other = __shfl_xor_sync(mask, value, step);
    value = other < value ? other : value;
  }
  return value;
}

template <typename MaskT, typename T>
__device__ __forceinline__ T __reduce_add_sync(MaskT mask, T value) {
  for (int step = 1; step < static_cast<int>(kLogicalWarpSize); step <<= 1)
    value += __shfl_xor_sync(mask, value, step);
  return value;
}

// Ballot of the caller's 32-lane logical warp.
template <typename MaskT>
__device__ __forceinline__ uint32_t __ballot_sync(MaskT, int predicate) {
  uint64_t wave_ballot = __ballot(predicate);
  uint32_t logical_warp_shift = humming_platform::hardware_lane() & kLogicalWarpSize;
  return static_cast<uint32_t>(wave_ballot >> logical_warp_shift);
}

template <typename MaskT>
__device__ __forceinline__ int __any_sync(MaskT mask, int predicate) {
  return __ballot_sync(mask, predicate) != 0;
}

template <typename MaskT>
__device__ __forceinline__ int __all_sync(MaskT mask, int predicate) {
  return __ballot_sync(mask, predicate) == 0xFFFFFFFFu;
}

__device__ __forceinline__ uint32_t __activemask() {
  uint64_t wave_ballot = __ballot(true);
  uint32_t logical_warp_shift = humming_platform::hardware_lane() & kLogicalWarpSize;
  return static_cast<uint32_t>(wave_ballot >> logical_warp_shift);
}

__device__ __forceinline__ void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

template <typename MaskT>
__device__ __forceinline__ void __syncwarp(MaskT) {
  __syncwarp();
}

// No named barriers on CDNA: see the contract at the top of this file.
#define HUMMING_NAMED_BARRIER_SYNC(barrier_id, num_threads) \
  do {                                                      \
    (void)(barrier_id);                                     \
    (void)(num_threads);                                    \
    __syncthreads();                                        \
  } while (0)

// The NVIDIA debug path reports through __assertfail; trap is the closest AMDGPU equivalent.
#define HUMMING_DEVICE_ASSERT_FAIL(message) __builtin_trap()

#endif  // HUMMING_HIP

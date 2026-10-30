#pragma once
// ROCm replacement for utils/ptx/warp.cuh, on 32-lane logical warps (see platform.cuh).

CUDA_INLINE bool warp_elect_leader() {
  const uint32_t logical_lane = static_cast<uint32_t>(humming_platform::hardware_lane()) % kLogicalWarpSize;
  return logical_lane == 0;
}

CUDA_INLINE uint32_t warp_reduce_add(uint32_t local_count) { return __reduce_add_sync(0xffffffff, local_count); }

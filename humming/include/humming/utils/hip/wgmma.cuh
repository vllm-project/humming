#pragma once
// ROCm replacement for utils/ptx/wgmma.cuh: no warpgroup MMA on CDNA.

#include <humming/utils/hip/unsupported.cuh>

HUMMING_ROCM_UNSUPPORTED_STUB(void, wgmma_fence)
HUMMING_ROCM_UNSUPPORTED_STUB(void, wgmma_commit)
HUMMING_ROCM_UNSUPPORTED_STUB(void, wgmma_wait)

// Register fences are a compiler-ordering hint; keep them usable from portable code.
CUDA_INLINE void warpgroup_fence_operand(uint32_t &reg) { asm volatile("" : "+v"(reg)::"memory"); }

CUDA_INLINE void warpgroup_fence_operand(float &reg) { asm volatile("" : "+v"(reg)::"memory"); }

#pragma once
// ROCm replacement for utils/ptx/tma.cuh: CDNA has no TMA, every entry point is a stub.

#include <humming/utils/hip/unsupported.cuh>

HUMMING_ROCM_UNSUPPORTED_STUB(uint64_t, create_tma_cache_policy)
HUMMING_ROCM_UNSUPPORTED_STUB(void, prefetch_tensor_map)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_load_1d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_load_2d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_load_3d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_load_5d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_prefetch_1d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_prefetch_2d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_prefetch_3d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_store_2d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_store_3d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_reduce_add_2d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_reduce_add_3d)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_expect_tx)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_commit_store_group)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_fence_async_shared)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tma_wait_store_group)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tensor_map_replace_global_dim)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tensor_map_release_cta)
HUMMING_ROCM_UNSUPPORTED_STUB(void, tensor_map_acquire_cta)

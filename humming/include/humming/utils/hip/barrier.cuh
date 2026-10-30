#pragma once
// ROCm replacement for utils/ptx/barrier.cuh.
//
// - griddepcontrol (PDL) does not exist on ROCm: no-ops, kernels are launched without PDL.
// - Partial-block barriers fall back to __syncthreads() (HUMMING_NAMED_BARRIER_SYNC), valid only
//   when every thread of the block takes the same sequence of syncs.
// - Stream-K global locks use HIP atomics with agent (device) scope.
// - mbarrier is NVIDIA-only: stubs.

#include <humming/utils/hip/unsupported.cuh>

CUDA_INLINE void griddepcontrol_wait() {}

CUDA_INLINE void griddepcontrol_launch_dependents() {}

template <uint32_t kNumSyncThreads, uint32_t kNumThreads, uint32_t kBarrierId = 1>
CUDA_INLINE void sync_part_threads(uint32_t barrier_id = kBarrierId) {
  if constexpr (kNumSyncThreads == kNumThreads) {
    __syncthreads();
  } else {
    static_assert(kNumThreads >= kNumSyncThreads);
    static_assert(kNumSyncThreads > 0);
    HUMMING_NAMED_BARRIER_SYNC(barrier_id, kNumSyncThreads);
  }
}

CUDA_INLINE int load_lock_acquire(int *lock) {
  return __hip_atomic_load(lock, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

CUDA_INLINE void add_lock_release(int *lock, int value) {
  __hip_atomic_fetch_add(lock, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_acquire(int *lock, int count, uint32_t thread_id = threadIdx.x, uint32_t barrier_id = 1) {
  if (thread_id == 0) {
#if HUMMING_DEBUG_KERNEL
    uint64_t start_clock = debug_kernel_timer_start();
#endif
    int state = -1;
    do {
      state = load_lock_acquire(lock);
#if HUMMING_DEBUG_KERNEL
      debug_kernel_timeout_check(start_clock, "Humming Stream-K barrier timeout");
#endif
    } while (state != count);
  }
  sync_part_threads<kNumSyncThreads, kNumThreads>(barrier_id);
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_acquire2(int *lock, int count, uint32_t thread_id = threadIdx.x, uint32_t barrier_id = 1) {
  if (thread_id == 0) {
#if HUMMING_DEBUG_KERNEL
    uint64_t start_clock = debug_kernel_timer_start();
#endif
    int state = 1;
    do {
      state = load_lock_acquire(lock);
#if HUMMING_DEBUG_KERNEL
      debug_kernel_timeout_check(start_clock, "Humming Stream-K barrier timeout");
#endif
    } while (state > count);
  }
  sync_part_threads<kNumSyncThreads, kNumThreads>(barrier_id);
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_release(int *lock, bool reset = false, uint32_t thread_id = threadIdx.x, uint32_t barrier_id = 1) {
  sync_part_threads<kNumSyncThreads, kNumThreads>(barrier_id);
  if (thread_id == 0) {
    if (reset) {
      __hip_atomic_store(lock, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      add_lock_release(lock, 1);
    }
  }
}

template <uint32_t kNumSyncThreads = 0, uint32_t kNumThreads = 0>
CUDA_INLINE void barrier_release2(int *lock, int32_t val, uint32_t thread_id = threadIdx.x, uint32_t barrier_id = 1) {
  sync_part_threads<kNumSyncThreads, kNumThreads>(barrier_id);
  if (thread_id == 0) {
    if (val < 0) {
      __hip_atomic_store(lock, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      add_lock_release(lock, 1);
    }
  }
}

HUMMING_ROCM_UNSUPPORTED_STUB(void, mbarrier_wait)
HUMMING_ROCM_UNSUPPORTED_STUB(void, mbarrier_init_sync)
HUMMING_ROCM_UNSUPPORTED_STUB(void, mbarrier_arrive)

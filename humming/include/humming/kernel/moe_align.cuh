#include <humming/utils/all.cuh>

// moe_align_block_size in one workgroup (see humming/ops/moe.py for the contract).
//
// 1. count the routed rows of every expert (LDS atomics),
// 2. inclusive scan of the per-expert block counts (Hillis-Steele over kNumThreads experts),
// 3. expert_ids[b] by binary search of the block end offsets,
// 4. scatter the row ids to their expert's padded segment (LDS atomic cursor per expert).
//
// The order of rows inside one expert is not deterministic (like vLLM's op); the GEMM does not
// depend on it. Requires num_experts <= kNumThreads.
//
// Cost on MI355X: ~3.5 us up to ~1k routed rows; one workgroup then becomes LDS-atomic / scattered
// store bound (~0.85 ns per routed row: 15 us for 16k rows, 55 us for 64k rows). Unrolling the
// loads (8 independent loads per thread) did not help, the loads are not the limit.
constexpr int32_t kMoeAlignThreads = 1024;

template <bool kInt64Ids>
__global__ void __launch_bounds__(kMoeAlignThreads) moe_align_kernel(
    const void *topk_ids_ptr, const int32_t *expert_map, int32_t *sorted_ids, int32_t *expert_ids,
    int32_t *num_tokens_padded, int32_t numel, int32_t num_experts, int32_t block_size,
    int32_t max_num_tokens_padded, int32_t max_num_blocks, int32_t ignore_invalid_experts) {
  __shared__ int32_t s_count[kMoeAlignThreads];
  __shared__ int32_t s_block_end[kMoeAlignThreads];
  __shared__ int32_t s_cursor[kMoeAlignThreads];

  const int32_t tid = threadIdx.x;
  const bool filter_by_map = expert_map != nullptr && ignore_invalid_experts;

  auto load_expert = [&](int32_t i) -> int32_t {
    int64_t e = kInt64Ids ? static_cast<const int64_t *>(topk_ids_ptr)[i]
                          : static_cast<int64_t>(static_cast<const int32_t *>(topk_ids_ptr)[i]);
    if (e < 0 || e >= num_experts) return -1;
    if (filter_by_map && expert_map[e] < 0) return -1;
    return static_cast<int32_t>(e);
  };

  s_count[tid] = 0;
  for (int32_t p = tid; p < max_num_tokens_padded; p += kMoeAlignThreads)
    sorted_ids[p] = numel;
  __syncthreads();

  for (int32_t i = tid; i < numel; i += kMoeAlignThreads) {
    const int32_t e = load_expert(i);
    if (e >= 0) atomicAdd(&s_count[e], 1);
  }
  __syncthreads();

  const int32_t my_blocks = tid < num_experts ? (s_count[tid] + block_size - 1) / block_size : 0;
  s_block_end[tid] = my_blocks;
  __syncthreads();
  for (int32_t offset = 1; offset < kMoeAlignThreads; offset <<= 1) {
    const int32_t add = tid >= offset ? s_block_end[tid - offset] : 0;
    __syncthreads();
    s_block_end[tid] += add;
    __syncthreads();
  }
  if (tid < num_experts) s_cursor[tid] = (s_block_end[tid] - my_blocks) * block_size;
  const int32_t total_blocks = s_block_end[num_experts - 1];
  if (tid == 0) num_tokens_padded[0] = total_blocks * block_size;

  for (int32_t b = tid; b < max_num_blocks; b += kMoeAlignThreads) {
    int32_t value = -1;
    if (b < total_blocks) {
      // First expert whose inclusive block end exceeds b.
      int32_t lo = 0, hi = num_experts - 1;
      while (lo < hi) {
        const int32_t mid = (lo + hi) >> 1;
        if (s_block_end[mid] > b) hi = mid;
        else lo = mid + 1;
      }
      value = expert_map != nullptr ? expert_map[lo] : lo;
    }
    expert_ids[b] = value;
  }
  // Orders the sorted_ids fill above before the scatter below (workgroup scope).
  __syncthreads();

  for (int32_t i = tid; i < numel; i += kMoeAlignThreads) {
    const int32_t e = load_expert(i);
    if (e >= 0) sorted_ids[atomicAdd(&s_cursor[e], 1)] = i;
  }
}

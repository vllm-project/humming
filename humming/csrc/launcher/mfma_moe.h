#pragma once

// ROCm launch path for the gfx950 MFMA MoE GEMM (humming/kernel/humming_mfma_moe.cuh).
//
// An MoE kernel is an MFMA kernel (MFMA_BLOCK_M, see mfma.h) that also carries MFMA_MOE_TYPE
// (1 = indexed, 2 = grouped_contiguous, 3 = grouped_masked, Humming's GemmType ids). Split-K
// workspace / tile counters are shared with the dense launcher.

#ifdef HUMMING_HIP

#include <optional>

#include "./mfma.h"

inline std::optional<uint32_t> read_mfma_moe_type(CubinReader &reader) {
  if (!reader.hasSymbol("MFMA_MOE_TYPE")) return std::nullopt;
  return reader.getUint32("MFMA_MOE_TYPE");
}

// Blocks that do real work, used for the split-K decision. Indexed: the routed rows of the
// non-empty experts, each expert padded to whole blocks (at most min(E, rows) experts are hit).
inline int64_t mfma_moe_estimated_m_blocks(int64_t routed_rows, int64_t num_experts, int64_t block_m, int64_t grid_m) {
  int64_t hit_experts = std::min(num_experts, routed_rows);
  int64_t blocks = (routed_rows + hit_experts * (block_m - 1)) / block_m;
  return std::max<int64_t>(1, std::min(grid_m, blocks));
}

inline Tensor launch_mfma_moe_kernel(
    const MfmaKernelData &data,
    uint32_t moe_type,
    CUfunction func,
    int64_t num_cus,
    Tensor a,
    const Tensor &b,
    const Tensor &bs,
    const std::optional<Tensor> &bs2_,
    const std::optional<Tensor> &as_,
    const std::optional<Tensor> &as2_,
    const std::optional<Tensor> &bzp_,
    const std::optional<Tensor> &bias_,
    std::optional<Tensor> c_,
    const std::optional<Tensor> &sorted_ids_,
    const std::optional<Tensor> &expert_ids_,
    const std::optional<Tensor> &num_tokens_padded_,
    const std::optional<Tensor> &expert_layout_,
    const std::optional<Tensor> &locks_,
    int64_t top_k,
    int64_t valid_shape_m,
    bool should_check_tensor) {
  ASSERT_CHECK(!bs2_.has_value() && !as_.has_value() && !as2_.has_value(),
               "MFMA MoE GEMM supports neither input scales nor secondary weight scales");
  const bool is_indexed = moe_type == 1;
  const bool is_masked = moe_type == 3;
  ASSERT_CHECK(moe_type >= 1 && moe_type <= 3, "unknown MFMA MoE type ", moe_type);
  if (is_indexed) {
    ASSERT_CHECK(sorted_ids_.has_value() && expert_ids_.has_value() && num_tokens_padded_.has_value(),
                 "indexed MoE GEMM needs sorted_ids, expert_ids and num_tokens_padded");
  } else {
    ASSERT_CHECK(expert_layout_.has_value(), "grouped MoE GEMM needs expert_layout");
  }

  int64_t dev = a.get_device();
  ScalarType a_dtype = dtype_id_to_tensor_dtype(data.a_dtype_id);
  ASSERT_CHECK(a.dim() == 2, "inputs must be 2D for the MFMA MoE GEMM");
  a = torch_contiguous(a);
  ASSERT_CHECK(b.dim() == 3, "MoE weight must be [E, N, K * bits / 32]");
  int64_t a_rows = a.size(0);
  int64_t shape_k = a.size(1);
  int64_t num_experts = b.size(0);
  int64_t shape_n = b.size(1);
  int64_t num_groups = data.group_size == 0 ? 1 : shape_k / data.group_size;
  if (!is_indexed) top_k = 1;
  ASSERT_CHECK(top_k >= 1, "top_k must be positive");
  int64_t c_rows = a_rows * top_k;

  Tensor c = c_.has_value() ? c_.value() : torch_empty({c_rows, shape_n}, a_dtype, a.device());
  ASSERT_CHECK(c.dim() == 2 && c.size(0) == c_rows && c.size(1) <= shape_n && c.stride(1) == 1,
               "outputs must be [", c_rows, ", <= ", shape_n, "] with unit column stride");
  int64_t n_out = c.size(1);
  int64_t ldc = c.stride(0);

  int64_t bias_stride = 0;
  if (bias_.has_value()) {
    const Tensor &bias = bias_.value();
    bias_stride = bias.size(-1);
    ASSERT_CHECK(bias.is_contiguous() && bias.scalar_type() == a_dtype && bias_stride >= n_out &&
                     bias.numel() == num_experts * bias_stride,
                 "MoE bias must be a contiguous [E, >= N] tensor of the activation dtype");
  }

  if (should_check_tensor) {
    ASSERT_CHECK(shape_k % 128 == 0, "MFMA GEMM needs K % 128 == 0, got ", shape_k);
    ASSERT_CHECK(shape_n % 64 == 0, "MFMA GEMM needs N % 64 == 0, got ", shape_n);
    ASSERT_CHECK(data.group_size == 0 || shape_k % data.group_size == 0, "K must be a multiple of the group size");
    ASSERT_CHECK(a_rows * shape_k * 2 < (int64_t(1) << 32), "MFMA MoE GEMM inputs must be < 4 GiB");
    check_mfma_tensor(a, "inputs", dev, a_dtype, a_rows, shape_k);
    ASSERT_CHECK(b.is_cuda() && b.get_device() == dev && b.is_contiguous() && b.scalar_type() == ScalarType::Int &&
                     b.size(2) == shape_k * data.num_bits / 32,
                 "weight must be a contiguous int32 [E, N, K * bits / 32] tensor");
    ASSERT_CHECK(bs.is_contiguous() && bs.scalar_type() == a_dtype && bs.numel() == num_experts * num_groups * shape_n,
                 "weight_scale must be a contiguous [E, groups, N] tensor of the activation dtype");
    if (data.has_zero_point) {
      ASSERT_CHECK(bzp_.has_value(), "zero_point is required by this kernel");
      ASSERT_CHECK(bzp_->is_contiguous() && bzp_->scalar_type() == ScalarType::Byte &&
                       bzp_->numel() == num_experts * num_groups * shape_n,
                   "zero_point must be a contiguous uint8 [E, groups, N] tensor");
    } else {
      ASSERT_CHECK(!bzp_.has_value(), "zero_point given to a kernel built without zero points");
    }
    ASSERT_CHECK(c.is_cuda() && c.get_device() == dev && c.scalar_type() == a_dtype, "bad outputs tensor");
    if (is_indexed) {
      for (const Tensor *t : {&sorted_ids_.value(), &expert_ids_.value(), &num_tokens_padded_.value()}) {
        ASSERT_CHECK(t->is_cuda() && t->get_device() == dev && t->is_contiguous() && t->scalar_type() == ScalarType::Int,
                     "sorted_ids / expert_ids / num_tokens_padded must be contiguous int32 device tensors");
      }
      ASSERT_CHECK(sorted_ids_->numel() >= expert_ids_->numel() * data.block_m,
                   "sorted_ids must cover expert_ids.numel() * block_m rows (moe_align_block_size with block_size=",
                   data.block_m, ")");
    } else {
      const Tensor &layout = expert_layout_.value();
      ASSERT_CHECK(layout.is_cuda() && layout.get_device() == dev && layout.is_contiguous() &&
                       (layout.scalar_type() == ScalarType::Int || layout.scalar_type() == ScalarType::Long),
                   "expert_layout must be a contiguous int32/int64 device tensor");
      ASSERT_CHECK(layout.numel() == num_experts + (is_masked ? 0 : 1), "expert_layout must have ",
                   num_experts + (is_masked ? 0 : 1), " entries");
      if (is_masked) ASSERT_CHECK(a_rows % num_experts == 0, "grouped_masked inputs must be [E * max_tokens, K]");
    }
  }
  if (a_rows == 0 || num_experts == 0) return c;

  int64_t grid_m;
  if (is_indexed) {
    grid_m = expert_ids_->numel();
  } else if (is_masked) {
    int64_t max_tokens = a_rows / num_experts;
    grid_m = std::min(num_experts * CEIL_DIV(max_tokens, data.block_m),
                      CEIL_DIV(a_rows, data.block_m) + std::min(num_experts, a_rows));
  } else {
    grid_m = CEIL_DIV(a_rows, data.block_m) + std::min(num_experts, a_rows);
  }
  if (grid_m == 0) return c;
  ASSERT_CHECK(grid_m < 65536, "MFMA MoE GEMM grid has too many M blocks (", grid_m, ")");
  int64_t grid_n = CEIL_DIV(shape_n, data.block_n);

  CUstream stream = get_current_cuda_stream(dev);
  int64_t routed_rows = valid_shape_m > 0 ? valid_shape_m : c_rows;
  int64_t active_m = mfma_moe_estimated_m_blocks(routed_rows, num_experts, data.block_m, grid_m);
  uint32_t num_splits = mfma_num_splits(data, active_m * data.block_m, shape_n, shape_k, num_cus);
  int64_t num_tiles = grid_n * grid_m;
  if (num_tiles > kMfmaFallbackCounters) num_splits = 1;
  void *workspace_ptr = nullptr;
  void *counters_ptr = nullptr;
  std::optional<Tensor> workspace;
  if (num_splits > 1) {
    workspace = get_mfma_workspace(a, stream, int64_t(num_splits) * num_tiles * data.block_m * data.block_n);
    workspace_ptr = workspace->data_ptr();
    counters_ptr = get_mfma_tile_counters(locks_, dev, stream, num_tiles);
  }

  const void *a_ptr = a.data_ptr();
  const void *b_ptr = b.data_ptr();
  const void *bs_ptr = bs.data_ptr();
  const void *bzp_ptr = data.has_zero_point ? bzp_->data_ptr() : nullptr;
  const void *bias_ptr = bias_.has_value() ? bias_->data_ptr() : nullptr;
  void *c_ptr = c.data_ptr();
  const void *sorted_ids_ptr = is_indexed ? sorted_ids_->data_ptr() : nullptr;
  const void *expert_ids_ptr = is_indexed ? expert_ids_->data_ptr() : nullptr;
  const void *num_tokens_padded_ptr = is_indexed ? num_tokens_padded_->data_ptr() : nullptr;
  const void *expert_layout_ptr = is_indexed ? nullptr : expert_layout_->data_ptr();
  uint32_t layout_is_int64 = !is_indexed && expert_layout_->scalar_type() == ScalarType::Long;
  uint32_t a_rows_u32 = static_cast<uint32_t>(a_rows);
  uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
  uint32_t experts_u32 = static_cast<uint32_t>(num_experts);
  uint32_t n_u32 = static_cast<uint32_t>(shape_n);
  uint32_t k_u32 = static_cast<uint32_t>(shape_k);
  uint32_t n_out_u32 = static_cast<uint32_t>(n_out);
  uint32_t ldc_u32 = static_cast<uint32_t>(ldc);
  uint32_t bias_stride_u32 = static_cast<uint32_t>(bias_stride);
  void *kernel_args[] = {&a_ptr, &b_ptr, &bs_ptr, &bzp_ptr, &bias_ptr, &c_ptr, &workspace_ptr, &counters_ptr,
                         &sorted_ids_ptr, &expert_ids_ptr, &num_tokens_padded_ptr, &expert_layout_ptr,
                         &layout_is_int64, &a_rows_u32, &top_k_u32, &experts_u32, &n_u32, &k_u32, &n_out_u32,
                         &ldc_u32, &bias_stride_u32};

  check_curesult(
      hipModuleLaunchKernel(
          func, static_cast<uint32_t>(grid_n), static_cast<uint32_t>(grid_m), num_splits, data.num_threads, 1, 1, 0,
          stream, kernel_args, nullptr),
      "hipModuleLaunchKernel");
  return c;
}

#endif  // HUMMING_HIP

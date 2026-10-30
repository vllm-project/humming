#pragma once

// ROCm launch path for the gfx950 low-bit activation GEMM (humming/kernel/humming_mfma_f8.cuh).
// Included by mfma.h after the shared split-K helpers; launch_mfma_kernel() forwards kernels that
// carry the MFMA_F8_MODE symbol here.
//
// Tensors (humming/kernel/humming_mfma_f8.py::transform_mfma_f8_tensors):
//   inputs        [M, K] FP8 / INT8, or [M, K / 2] uint8 (packed FP4)
//   weight        [N, K * bits / 32] int32
//   weight_scale  FP32 [K / ws_group, N] (flush), uint8 / E8M0 [K / ws_group, N], or unused
//   weight_scale_2  FP32 [N]: epilogue scale per column
//   input_scale   FP32 [1] (tensor) or [M, 1] (token) or [M, >= K / as_group] (group), or E8M0
//                 bytes per row (uint8 / E8M0 / int32-packed) for E8M0 group scales
//   zero_point    uint8 [K / zp_group or 1, N]

#ifdef HUMMING_HIP

constexpr uint32_t kMfmaF8ScaleModeNone = 0;
constexpr uint32_t kMfmaF8ScaleModeFlush = 1;
constexpr int64_t kMfmaF8Int8MaxK = 131072;

inline Tensor launch_mfma_f8_kernel(
    const MfmaKernelData &data,
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
    const std::optional<Tensor> &locks_,
    bool should_check_tensor) {
  const MfmaF8Fields &f8 = data.f8;
  ASSERT_CHECK(!as2_.has_value(), "MFMA low-bit GEMM does not support secondary input scales");
  ASSERT_CHECK(!bias_.has_value(), "MFMA GEMM has no bias epilogue; add the bias outside the kernel");
  ASSERT_CHECK(!sorted_ids_.has_value() && !expert_ids_.has_value(), "MFMA GEMM does not support MoE");
  ASSERT_CHECK(bs2_.has_value(), "MFMA low-bit GEMM needs weight_scale_2 (FP32 [N] epilogue scale)");
  ASSERT_CHECK(as_.has_value(), "MFMA low-bit GEMM needs input_scale (quantize the input first)");
  const Tensor &bs2 = bs2_.value();
  const Tensor &as = as_.value();

  int64_t dev = a.get_device();
  ScalarType a_dtype = dtype_id_to_tensor_dtype(data.a_dtype_id);
  ScalarType c_dtype = dtype_id_to_tensor_dtype(f8.c_dtype_id);
  ASSERT_CHECK(a.dim() == 2, "inputs must be 2D for the MFMA GEMM");
  a = torch_contiguous(a);
  int64_t shape_m = a.size(0);
  int64_t shape_k = a.size(1) * f8.a_pack;
  int64_t shape_n = b.size(0);

  // Activation scale stride per row: FP32 elements (none / flush) or bytes (E8M0 modes).
  bool as_is_fp32 = f8.as_mode == kMfmaF8ScaleModeNone || f8.as_mode == kMfmaF8ScaleModeFlush;
  int64_t as_stride = 0;
  if (as.numel() != 1) {
    ASSERT_CHECK(as.dim() == 2 && as.size(0) == shape_m, "input_scale must have shape [M, ...], got ",
                 as.dim() == 2 ? as.size(0) : -1, " rows for M = ", shape_m);
    ASSERT_CHECK(as.stride(1) == 1, "input_scale rows must be contiguous");
    as_stride = as.stride(0) * static_cast<int64_t>(as.element_size()) / (as_is_fp32 ? 4 : 1);
  }

  Tensor c = c_.has_value() ? c_.value() : torch_empty({shape_m, shape_n}, c_dtype, a.device());
  if (should_check_tensor) {
    ASSERT_CHECK(shape_k % 128 == 0, "MFMA GEMM needs K % 128 == 0, got ", shape_k);
    ASSERT_CHECK(shape_n % 64 == 0, "MFMA GEMM needs N % 64 == 0, got ", shape_n);
    ASSERT_CHECK(!f8.is_int8 || shape_k <= kMfmaF8Int8MaxK, "INT8 MFMA GEMM supports K <= ", kMfmaF8Int8MaxK,
                 " (INT32 accumulator), got ", shape_k);
    check_mfma_tensor(a, "inputs", dev, a_dtype, shape_m, shape_k / f8.a_pack);
    check_mfma_tensor(b, "weight", dev, ScalarType::Int, shape_n, shape_k * data.num_bits / 32);
    if (f8.ws_mode != kMfmaF8ScaleModeNone) {
      int64_t num_groups = shape_k / f8.ws_group;
      ASSERT_CHECK(shape_k % f8.ws_group == 0, "K must be a multiple of the weight scale group size");
      if (f8.ws_mode == kMfmaF8ScaleModeFlush) {
        check_mfma_tensor(bs, "weight_scale", dev, ScalarType::Float, num_groups, shape_n);
      } else {
        ASSERT_CHECK(bs.element_size() == 1, "E8M0 weight_scale must have 1-byte elements");
        check_mfma_tensor(bs, "weight_scale", dev, bs.scalar_type(), num_groups, shape_n);
      }
    }
    ASSERT_CHECK(bs2.is_cuda() && bs2.get_device() == dev && bs2.is_contiguous() &&
                     bs2.scalar_type() == ScalarType::Float && bs2.numel() >= shape_n,
                 "weight_scale_2 must be a contiguous FP32 [N] tensor on the input device");
    ASSERT_CHECK(as.is_cuda() && as.get_device() == dev, "input_scale must be on the device of the input");
    if (as_is_fp32) {
      ASSERT_CHECK(as.scalar_type() == ScalarType::Float, "input_scale must be FP32, got ",
                   DTYPE_TO_STRING(as.scalar_type()));
    }
    if (f8.as_mode != kMfmaF8ScaleModeNone) {
      ASSERT_CHECK(as.numel() != 1 && shape_k % f8.as_group == 0, "group input_scale expected");
      int64_t row_units = as.size(1) * static_cast<int64_t>(as.element_size()) / (as_is_fp32 ? 4 : 1);
      ASSERT_CHECK(row_units >= shape_k / f8.as_group, "input_scale has ", row_units, " groups per row, expected ",
                   shape_k / f8.as_group);
    } else {
      ASSERT_CHECK(as.numel() == 1 || as.size(1) == 1, "per-token input_scale must have shape [M, 1]");
    }
    if (data.has_zero_point) {
      ASSERT_CHECK(bzp_.has_value(), "zero_point is required by this kernel");
      int64_t num_zp_groups = f8.zp_group == 0 ? 1 : shape_k / f8.zp_group;
      check_mfma_tensor(bzp_.value(), "zero_point", dev, ScalarType::Byte, num_zp_groups, shape_n);
    } else {
      ASSERT_CHECK(!bzp_.has_value(), "zero_point given to a kernel built without zero points");
    }
    check_mfma_tensor(c, "outputs", dev, c_dtype, shape_m, shape_n);
  }
  if (shape_m == 0) return c;

  CUstream stream = get_current_cuda_stream(dev);
  uint32_t num_splits = mfma_num_splits(data, shape_m, shape_n, shape_k, num_cus);
  int64_t num_tiles = CEIL_DIV(shape_n, data.block_n) * CEIL_DIV(shape_m, data.block_m);
  void *workspace_ptr = nullptr;
  void *counters_ptr = nullptr;
  std::optional<Tensor> workspace;  // keeps a capture-time allocation alive until the launch is recorded
  if (num_splits > 1) {
    workspace = get_mfma_workspace(a, stream, int64_t(num_splits) * num_tiles * data.block_m * data.block_n);
    workspace_ptr = workspace->data_ptr();
    counters_ptr = get_mfma_tile_counters(locks_, dev, stream, num_tiles);
  }

  const void *a_ptr = a.data_ptr();
  const void *b_ptr = b.data_ptr();
  const void *bs_ptr = bs.data_ptr();
  const void *bzp_ptr = data.has_zero_point ? bzp_->data_ptr() : nullptr;
  const void *bs2_ptr = bs2.data_ptr();
  const void *as_ptr = as.data_ptr();
  uint32_t as_stride_u32 = static_cast<uint32_t>(as_stride);
  void *c_ptr = c.data_ptr();
  uint32_t m_u32 = static_cast<uint32_t>(shape_m);
  uint32_t n_u32 = static_cast<uint32_t>(shape_n);
  uint32_t k_u32 = static_cast<uint32_t>(shape_k);
  void *kernel_args[] = {&a_ptr, &b_ptr, &bs_ptr, &bzp_ptr, &bs2_ptr, &as_ptr, &as_stride_u32,
                         &c_ptr, &workspace_ptr, &counters_ptr, &m_u32, &n_u32, &k_u32};

  check_curesult(
      hipModuleLaunchKernel(
          func,
          static_cast<uint32_t>(CEIL_DIV(shape_n, data.block_n)),
          static_cast<uint32_t>(CEIL_DIV(shape_m, data.block_m)),
          num_splits,
          data.num_threads, 1, 1,
          0, stream, kernel_args, nullptr),
      "hipModuleLaunchKernel");
  return c;
}

#endif  // HUMMING_HIP

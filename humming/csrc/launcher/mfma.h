#pragma once

// ROCm launch path for the gfx950 MFMA weight-only GEMM (humming/kernel/humming_mfma.cuh).
//
// The MFMA kernel has its own argument list and grid (N tiles x M tiles x K splits), so it does
// not go through the CUDA kernel's KernelData / TMA plumbing. register_kernel() recognises it by
// the MFMA_BLOCK_M symbol and launch_kernel() dispatches here; the torch op schema is unchanged.

#ifdef HUMMING_HIP

#include <map>
#include <mutex>
#include <optional>
#include <utility>

#include "./elf.h"
#include "./gpu_api.h"
#include "./torch_api.h"
#include "./utils.h"

// Low-bit activation kernels (humming/kernel/humming_mfma_f8.cuh) export MFMA_F8_* symbols on top
// of the MFMA_* ones and are launched by launch_mfma_f8_kernel (mfma_f8.h).
struct MfmaF8Fields {
  bool enabled = false;
  uint32_t c_dtype_id = 0;
  uint32_t a_pack = 1;  // K values per activation byte (2 for packed FP4)
  bool is_int8 = false;
  uint32_t ws_mode = 0;
  uint32_t ws_group = 0;
  uint32_t zp_group = 0;
  uint32_t as_mode = 0;
  uint32_t as_group = 0;
};

struct MfmaKernelData {
  uint32_t block_m;
  uint32_t block_n;
  uint32_t num_threads;
  uint32_t a_dtype_id;
  uint32_t num_bits;
  uint32_t group_size;  // 0 = per-channel scale
  bool has_zero_point;
  uint32_t max_splits;
  MfmaF8Fields f8{};
};

inline bool is_mfma_cubin(const CubinReader &reader) {
  return reader.hasSymbol("MFMA_BLOCK_M");
}

inline MfmaKernelData read_mfma_kernel_data(CubinReader &reader) {
  MfmaKernelData data{
      reader.getUint32("MFMA_BLOCK_M"),
      reader.getUint32("MFMA_BLOCK_N"),
      reader.getUint32("MFMA_NUM_THREADS"),
      reader.getUint32("MFMA_A_DTYPE_ID"),
      reader.getUint32("MFMA_NUM_BITS"),
      reader.getUint32("MFMA_GROUP_SIZE"),
      reader.getBool("MFMA_HAS_ZERO_POINT"),
      reader.getUint32("MFMA_MAX_SPLITS"),
  };
  if (reader.hasSymbol("MFMA_F8_MODE")) {
    data.f8.enabled = true;
    data.f8.c_dtype_id = reader.getUint32("MFMA_F8_C_DTYPE_ID");
    data.f8.a_pack = reader.getUint32("MFMA_F8_A_PACK");
    data.f8.is_int8 = reader.getBool("MFMA_F8_IS_INT8");
    data.f8.ws_mode = reader.getUint32("MFMA_F8_WS_MODE");
    data.f8.ws_group = reader.getUint32("MFMA_F8_WS_GROUP");
    data.f8.zp_group = reader.getUint32("MFMA_F8_ZP_GROUP");
    data.f8.as_mode = reader.getUint32("MFMA_F8_AS_MODE");
    data.f8.as_group = reader.getUint32("MFMA_F8_AS_GROUP");
  }
  return data;
}

// Split-K: keep doubling until about one block per CU, each split keeping >= 4 K chunks of 128.
// Mirrors humming/tune/gfx950.py::get_num_splits.
inline uint32_t mfma_num_splits(const MfmaKernelData &data, int64_t shape_m, int64_t shape_n, int64_t shape_k, int64_t num_cus) {
  int64_t num_blocks = CEIL_DIV(shape_n, data.block_n) * CEIL_DIV(shape_m, data.block_m);
  int64_t num_chunks_k = shape_k / 128;
  uint32_t num_splits = 1;
  while (num_blocks * num_splits < num_cus && num_splits < data.max_splits && num_chunks_k / (num_splits * 2) >= 4) {
    num_splits *= 2;
  }
  return num_splits;
}

inline bool mfma_stream_is_capturing(CUstream stream) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  check_curesult(hipStreamIsCapturing(stream, &status), "hipStreamIsCapturing");
  return status != hipStreamCaptureStatusNone;
}

// Split-K FP32 partials. Outside stream capture a grow-only buffer per (device, stream) is reused:
// launches on one stream are ordered, and a buffer is only released through the caching allocator
// on the stream that allocated it. Under capture every launch gets its own allocation from the
// graph's private pool, so a later grow of the eager buffer can never free memory a graph uses.
inline Tensor get_mfma_workspace(const Tensor &a, CUstream stream, int64_t num_floats) {
  if (mfma_stream_is_capturing(stream)) return torch_empty({num_floats}, ScalarType::Float, a.device());

  static std::mutex mutex;
  // Leaked on purpose: tensors must not be destroyed after the HIP runtime at process exit.
  static auto *workspaces = new std::map<std::pair<int64_t, CUstream>, Tensor>();
  std::lock_guard<std::mutex> lock(mutex);
  auto key = std::make_pair(static_cast<int64_t>(a.get_device()), stream);
  auto it = workspaces->find(key);
  if (it != workspaces->end() && it->second.numel() >= num_floats) return it->second;
  int64_t size = std::max<int64_t>(num_floats, int64_t(1) << 20);
  Tensor workspace = torch_empty({size}, ScalarType::Float, a.device());
  (*workspaces)[key] = workspace;
  return workspace;
}

// Tile counters must be zero before the launch; the last block of every tile resets its counter.
// The layer's `locks` buffer is used when it is large enough, otherwise a per-device buffer that
// is allocated (and zeroed) once, outside stream capture.
constexpr int64_t kMfmaFallbackCounters = 1 << 16;

inline void *get_mfma_tile_counters(const std::optional<Tensor> &locks_, int64_t dev, CUstream stream, int64_t num_tiles) {
  if (locks_.has_value()) {
    const Tensor &locks = locks_.value();
    bool usable = locks.is_contiguous() && locks.get_device() == dev && locks.numel() >= num_tiles &&
        (locks.scalar_type() == ScalarType::Int || locks.scalar_type() == ScalarType::UInt32);
    if (usable) return locks.data_ptr();
  }
  ASSERT_CHECK(num_tiles <= kMfmaFallbackCounters, "MFMA split-K needs ", num_tiles, " tile counters");

  static std::mutex mutex;
  static auto *buffers = new std::map<int64_t, void *>();
  std::lock_guard<std::mutex> lock(mutex);
  auto it = buffers->find(dev);
  if (it != buffers->end()) return it->second;
  ASSERT_CHECK(!mfma_stream_is_capturing(stream),
               "MFMA split-K tile counters are allocated on first use; pass `locks` (HummingLayer.locks) "
               "or run the layer once before graph capture");
  void *counters = nullptr;
  check_curesult(hipMalloc(&counters, kMfmaFallbackCounters * sizeof(uint32_t)), "hipMalloc");
  check_curesult(hipMemset(counters, 0, kMfmaFallbackCounters * sizeof(uint32_t)), "hipMemset");
  (*buffers)[dev] = counters;
  return counters;
}

inline void check_mfma_tensor(const Tensor &tensor, const char *name, int64_t dev, ScalarType dtype, int64_t rows, int64_t cols) {
  ASSERT_CHECK(tensor.is_cuda() && tensor.get_device() == dev, name, " must be on the device of the input");
  ASSERT_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  ASSERT_CHECK(tensor.scalar_type() == dtype, name, " has dtype ", DTYPE_TO_STRING(tensor.scalar_type()),
               ", expected ", DTYPE_TO_STRING(dtype));
  ASSERT_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == cols, name, " must have shape [",
               rows, ", ", cols, "]");
}

#include "./mfma_f8.h"

inline Tensor launch_mfma_kernel(
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
  if (data.f8.enabled) {
    return launch_mfma_f8_kernel(data, func, num_cus, a, b, bs, bs2_, as_, as2_, bzp_, bias_, c_, sorted_ids_,
                                 expert_ids_, locks_, should_check_tensor);
  }
  ASSERT_CHECK(!bs2_.has_value() && !as_.has_value() && !as2_.has_value(),
               "MFMA GEMM supports neither input scales nor secondary weight scales");
  ASSERT_CHECK(!bias_.has_value(), "MFMA GEMM has no bias epilogue; add the bias outside the kernel");
  ASSERT_CHECK(!sorted_ids_.has_value() && !expert_ids_.has_value(), "MFMA GEMM does not support MoE");

  int64_t dev = a.get_device();
  ScalarType a_dtype = dtype_id_to_tensor_dtype(data.a_dtype_id);
  ASSERT_CHECK(a.dim() == 2, "inputs must be 2D for the MFMA GEMM");
  a = torch_contiguous(a);
  int64_t shape_m = a.size(0);
  int64_t shape_k = a.size(1);
  int64_t shape_n = b.size(0);
  int64_t num_groups = data.group_size == 0 ? 1 : shape_k / data.group_size;

  Tensor c = c_.has_value() ? c_.value() : torch_empty({shape_m, shape_n}, a_dtype, a.device());
  if (should_check_tensor) {
    ASSERT_CHECK(shape_k % 128 == 0, "MFMA GEMM needs K % 128 == 0, got ", shape_k);
    ASSERT_CHECK(shape_n % 64 == 0, "MFMA GEMM needs N % 64 == 0, got ", shape_n);
    ASSERT_CHECK(data.group_size == 0 || shape_k % data.group_size == 0, "K must be a multiple of the group size");
    check_mfma_tensor(a, "inputs", dev, a_dtype, shape_m, shape_k);
    check_mfma_tensor(b, "weight", dev, ScalarType::Int, shape_n, shape_k * data.num_bits / 32);
    check_mfma_tensor(bs, "weight_scale", dev, a_dtype, num_groups, shape_n);
    if (data.has_zero_point) {
      ASSERT_CHECK(bzp_.has_value(), "zero_point is required by this kernel");
      check_mfma_tensor(bzp_.value(), "zero_point", dev, ScalarType::Byte, num_groups, shape_n);
    } else {
      ASSERT_CHECK(!bzp_.has_value(), "zero_point given to a kernel built without zero points");
    }
    check_mfma_tensor(c, "outputs", dev, a_dtype, shape_m, shape_n);
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
  void *c_ptr = c.data_ptr();
  uint32_t m_u32 = static_cast<uint32_t>(shape_m);
  uint32_t n_u32 = static_cast<uint32_t>(shape_n);
  uint32_t k_u32 = static_cast<uint32_t>(shape_k);
  void *kernel_args[] = {&a_ptr, &b_ptr, &bs_ptr, &bzp_ptr, &c_ptr, &workspace_ptr, &counters_ptr, &m_u32, &n_u32, &k_u32};

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

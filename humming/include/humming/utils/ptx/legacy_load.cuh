#pragma once

#include <humming/utils/base.cuh>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define CP_ASYNC_L2_HINT ".L2::256B"
#else
#define CP_ASYNC_L2_HINT ""
#endif
#define CP_ASYNC_PREFIX(MODE) "cp.async." #MODE ".shared.global" CP_ASYNC_L2_HINT

#define CP_ASYNC_ASM(MODE)                                          \
  asm volatile(CP_ASYNC_PREFIX(MODE) " [%0], [%1], %2;\n" \
               :                                                    \
               : "r"(smem), "l"(gmem_ptr), "n"(BYTES)               \
               : "memory");

#define CP_ASYNC_ZFILL_ASM(MODE)                                        \
  asm volatile(CP_ASYNC_PREFIX(MODE) " [%0], [%1], %2, %3;\n" \
               :                                                        \
               : "r"(smem), "l"(gmem_ptr), "n"(BYTES), "r"(src_size)    \
               : "memory");

#define CP_ASYNC_PRED_ASM(MODE)                                            \
  asm volatile("{\n"                                                       \
               "  .reg .pred p;\n"                                         \
               "  setp.ne.s32 p, %0, 0;\n"                                 \
               "  @p " CP_ASYNC_PREFIX(MODE) " [%1], [%2], %3;\n"       \
               "}\n"                                                       \
               :                                                           \
               : "r"((uint32_t)pred), "r"(smem), "l"(gmem_ptr), "n"(BYTES) \
               : "memory");

#define CP_ASYNC_ZFILL_PRED_ASM(MODE)                                                      \
  asm volatile("{\n"                                                                       \
               "  .reg .pred p;\n"                                                         \
               "  setp.ne.s32 p, %0, 0;\n"                                                 \
               "  @p " CP_ASYNC_PREFIX(MODE) " [%1], [%2], %3, %4;\n"                       \
               "}\n"                                                                       \
               :                                                                           \
               : "r"((uint32_t)pred2), "r"(smem), "l"(gmem_ptr), "n"(BYTES), "r"(src_size) \
               : "memory");


template <bool use_cp_async, typename T, uint32_t cache_type = 0>
CUDA_INLINE void legacy_load(const T *gmem_ptr, T *smem_ptr) {
  if constexpr (!use_cp_async) {
    smem_ptr[0] = gmem_ptr[0];
    return;
  }

  constexpr uint32_t BYTES = sizeof(T);
  static_assert(cache_type != 2 || BYTES == 16);
  uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);

  if constexpr (cache_type == 2 || (cache_type == 0 && BYTES == 16)) {
    CP_ASYNC_ASM(cg);
  } else {
    CP_ASYNC_ASM(ca);
  }
};

template <bool use_cp_async, typename T, uint32_t cache_type = 0>
CUDA_INLINE void legacy_load_pred(const T *gmem_ptr, T *smem_ptr, bool pred) {
  if constexpr (!use_cp_async) {
    if (pred) smem_ptr[0] = gmem_ptr[0];
    return;
  }

  constexpr uint32_t BYTES = sizeof(T);
  static_assert(cache_type != 2 || BYTES == 16);
  uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
  if constexpr (cache_type == 2 || (cache_type == 0 && BYTES == 16)) {
    CP_ASYNC_PRED_ASM(cg);
  } else {
    CP_ASYNC_PRED_ASM(ca);
  }
};

template <bool use_cp_async, typename T, uint32_t cache_type = 0>
CUDA_INLINE void legacy_load_zfill(const T *gmem_ptr, T *smem_ptr, bool pred) {
  if constexpr (!use_cp_async) {
    smem_ptr[0] = pred ? gmem_ptr[0] : T();
    return;
  }

  constexpr uint32_t BYTES = sizeof(T);
  static_assert(cache_type != 2 || BYTES == 16);
  uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
  const uint32_t src_size = pred ? BYTES : 0;
  if constexpr (cache_type == 2 || (cache_type == 0 && BYTES == 16)) {
    CP_ASYNC_ZFILL_ASM(cg);
  } else {
    CP_ASYNC_ZFILL_ASM(ca);
  }
};

template <bool use_cp_async, typename T, uint32_t cache_type = 0>
CUDA_INLINE void legacy_load_zfill_pred(const T *gmem_ptr, T *smem_ptr, bool pred1, bool pred2) {
  if constexpr (!use_cp_async) {
    const T val = {0, 0, 0, 0};
    if (pred2) { smem_ptr[0] = pred1 ? gmem_ptr[0] : val; }
    return;
  }

  constexpr uint32_t BYTES = sizeof(T);
  static_assert(cache_type != 2 || BYTES == 16);
  uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
  const uint32_t src_size = pred1 ? BYTES : 0;
  if constexpr (cache_type == 2 || (cache_type == 0 && BYTES == 16)) {
    CP_ASYNC_ZFILL_PRED_ASM(cg);
  } else {
    CP_ASYNC_ZFILL_PRED_ASM(ca);
  }
};

template <bool use_cp_async, uint32_t num_int4s, uint32_t threads, uint32_t thread_offset = 0, typename T = int4>
CUDA_INLINE void legacy_load_1d(const T *gmem_ptr, T *smem_ptr) {
  constexpr uint32_t iters = CEIL_DIV(num_int4s, threads);
  const uint32_t thread_id = threadIdx.x - thread_offset;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < iters; i++) {
    const uint32_t index = i * threads + thread_id;

    if (num_int4s % threads == 0 || i != iters - 1 || index < num_int4s) {
      legacy_load<use_cp_async>(gmem_ptr + index, smem_ptr + index);
    }
  }
};

template <bool use_cp_async, uint32_t num_int4s, uint32_t threads, uint32_t gmem_stride, uint32_t smem_stride, uint32_t thread_offset = 0, typename T = int4>
CUDA_INLINE void legacy_load_2d(const T *gmem_ptr, T *smem_ptr) {
  static_assert(num_int4s % smem_stride == 0);
  const uint32_t thread_id = threadIdx.x - thread_offset;

  if constexpr (smem_stride % threads == 0 || num_int4s < smem_stride) {
    constexpr uint32_t line_iters = smem_stride / threads;
    constexpr uint32_t num_lines = num_int4s / smem_stride;

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < num_lines; i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < line_iters; j++) {
        uint32_t col = j * threads + thread_id;
        uint32_t smem_offset = (i * line_iters + j) * threads + thread_id;
        uint32_t gmem_offset = i * gmem_stride + j * threads + thread_id;

        legacy_load<use_cp_async>(gmem_ptr + gmem_offset, smem_ptr + smem_offset);
      }
    }
  } else {
    constexpr uint32_t iters = CEIL_DIV(num_int4s, threads);

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < iters; i++) {
      uint32_t smem_offset = i * threads + thread_id;
      uint32_t gmem_row = smem_offset / smem_stride;
      uint32_t gmem_col = smem_offset - gmem_row * smem_stride;
      uint32_t gmem_offset = smem_offset + (gmem_stride - smem_stride) * gmem_row;

      if (num_int4s % threads == 0 || i != iters - 1) {
        legacy_load<use_cp_async>(gmem_ptr + gmem_offset, smem_ptr + smem_offset);
      } else {
        legacy_load_pred<use_cp_async>(gmem_ptr + gmem_offset, smem_ptr + smem_offset, smem_offset < num_int4s);
      }
    }
  }
}


CUDA_INLINE void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n");
}

CUDA_INLINE void cp_async_commit_mbarrier(uint64_t *smem_addr) {
  uint32_t smem = cast_smem_ptr_to_uint(smem_addr);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
               :
               : "r"(smem)
               : "memory");
}

template <uint32_t N>
CUDA_INLINE void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
};

template <uint32_t kBits>
CUDA_INLINE void aiu_load_gmem(
    const void *gmem_ptr, void *smem_ptr,
    uint32_t gmem_dim1, uint32_t gmem_dim2,
    uint32_t gmem_offset1, uint32_t gmem_offset2,
    uint32_t smem_dim1, uint32_t smem_dim2) {
  uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
  if constexpr (kBits == 16) {
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.stride.2d.b16 "
        "[%0], [%1], {1, %2, %3, 0, %4, %5}, {1, %6, %7, 1};"
        :
        : "r"(smem), "l"(gmem_ptr), "r"(gmem_dim1), "r"(gmem_dim2),
          "r"(gmem_offset1), "r"(gmem_offset2), "r"(smem_dim1), "r"(smem_dim2));
  } else if constexpr (kBits == 8) {
    asm volatile(
        "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.stride.2d.b8 "
        "[%0], [%1], {0, %2, %3, 0, %4, %5}, {0, %6, %7, 1};"
        :
        : "r"(smem), "l"(gmem_ptr), "r"(gmem_dim1), "r"(gmem_dim2),
          "r"(gmem_offset1), "r"(gmem_offset2), "r"(smem_dim1), "r"(smem_dim2));
  }
};


CUDA_INLINE void aiu_load_gmem_linear_2d(
    const void *gmem_ptr, void *smem_ptr,
    uint32_t gmem_dim1, uint32_t gmem_dim2,
    uint32_t gmem_offset1, uint32_t gmem_offset2,
    uint32_t smem_dim1, uint32_t smem_dim2) {
  uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.stride.2d.b32 "
      "[%0], [%1], {1, %2, %3, 0, %4, %5}, {1, %6, %7, 1};"
      :
      : "r"(smem), "l"(gmem_ptr), "r"(gmem_dim1), "r"(gmem_dim2),
        "r"(gmem_offset1), "r"(gmem_offset2), "r"(smem_dim1), "r"(smem_dim2));
};


CUDA_INLINE void aiu_load_gmem_linear_3d(
    const void *gmem_ptr, void *smem_ptr,
    uint32_t gmem_dim1, uint32_t gmem_dim2, uint32_t gmem_dim3,
    uint32_t gmem_offset1, uint32_t gmem_offset2, uint32_t gmem_offset3,
    uint32_t smem_dim1, uint32_t smem_dim2, uint32_t smem_dim3) {
  uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.padz.linear.stride.3d.b32 "
      "[%0], [%1], {%2, %3, %4, %5, %6, %7}, {%8, %9, %10, 1};"
      :
      : "r"(smem), "l"(gmem_ptr),
        "r"(gmem_dim1), "r"(gmem_dim2), "r"(gmem_dim3),
        "r"(gmem_offset1), "r"(gmem_offset2), "r"(gmem_offset3),
        "r"(smem_dim1), "r"(smem_dim2), "r"(smem_dim3));
};

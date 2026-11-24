#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <humming/utils/base.cuh>


CUDA_INLINE half2 add_f162(const half2 &a, const half2 &b) {
  return __hadd2(a, b);
}

CUDA_INLINE half2 sub_f162(const half2 &a, const half2 &b) {
  return __hsub2(a, b);
}

CUDA_INLINE half2 mul_f162(const half2 &a, const half2 &b) {
  return __hmul2(a, b);
}

CUDA_INLINE half2 fma_f162(const half2 &a, const half2 &b, const half2 &c) {
  return __hfma2(a, b, c);
}


// Reference parameters avoid ARM64 NVRTC 12.x miscompilation of bf16x2 value parameters.
CUDA_INLINE nv_bfloat162 add_f162(const nv_bfloat162 &a, const nv_bfloat162 &b) {
  nv_bfloat162 result;
#if __CUDA_ARCH__ >= 900
  asm("add.bf16x2 %0, %1, %2;"
      : "=r"(reinterpret_cast<uint32_t &>(result))
      : "r"(reinterpret_cast<const uint32_t &>(a)), "r"(reinterpret_cast<const uint32_t &>(b)));
#else
  asm("fma.rn.bf16x2 %0, %1, %3, %2;"
      : "=r"(reinterpret_cast<uint32_t &>(result))
      : "r"(reinterpret_cast<const uint32_t &>(a)), "r"(reinterpret_cast<const uint32_t &>(b)), "r"(0x3f803f80u));
#endif
  return result;
}

CUDA_INLINE nv_bfloat162 sub_f162(const nv_bfloat162 &a, const nv_bfloat162 &b) {
  nv_bfloat162 result;
#if __CUDA_ARCH__ >= 900
  asm("sub.bf16x2 %0, %1, %2;"
      : "=r"(reinterpret_cast<uint32_t &>(result))
      : "r"(reinterpret_cast<const uint32_t &>(a)), "r"(reinterpret_cast<const uint32_t &>(b)));
#else
  asm("fma.rn.bf16x2 %0, %2, %3, %1;"
      : "=r"(reinterpret_cast<uint32_t &>(result))
      : "r"(reinterpret_cast<const uint32_t &>(a)), "r"(reinterpret_cast<const uint32_t &>(b)), "r"(0xbf80bf80u));
#endif
  return result;
}

CUDA_INLINE nv_bfloat162 mul_f162(const nv_bfloat162 &a, const nv_bfloat162 &b) {
  nv_bfloat162 result;
#if __CUDA_ARCH__ >= 900
  asm("mul.bf16x2 %0, %1, %2;"
      : "=r"(reinterpret_cast<uint32_t &>(result))
      : "r"(reinterpret_cast<const uint32_t &>(a)), "r"(reinterpret_cast<const uint32_t &>(b)));
#else
  asm("fma.rn.bf16x2 %0, %1, %2, %3;"
      : "=r"(reinterpret_cast<uint32_t &>(result))
      : "r"(reinterpret_cast<const uint32_t &>(a)), "r"(reinterpret_cast<const uint32_t &>(b)), "r"(0x80008000u));
#endif
  return result;
}

CUDA_INLINE nv_bfloat162 fma_f162(const nv_bfloat162 &a, const nv_bfloat162 &b, const nv_bfloat162 &c) {
  nv_bfloat162 result;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;"
      : "=r"(reinterpret_cast<uint32_t &>(result))
      : "r"(reinterpret_cast<const uint32_t &>(a)), "r"(reinterpret_cast<const uint32_t &>(b)), "r"(reinterpret_cast<const uint32_t &>(c)));
  return result;
}

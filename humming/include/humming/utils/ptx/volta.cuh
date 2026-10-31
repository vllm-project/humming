#pragma once

// Volta (sm_70) stand-ins for two Turing instructions the kernels use: ldmatrix (tile load into the tensor-core
// fragment layout) and mma.sync m16n8k8 (the fp16 tensor-core multiply). Each takes the same registers as the
// instruction it replaces, so the loaders, dequantization and scales are unchanged. Checked against a CPU model of
// both instructions in results/volta_test.cu and results/volta_test2.cu.
//
// Volta's own tensor-core instruction is mma.sync.m8n8k4. Measured on a V100 (results/discover_m8n8k4.cu), one
// instruction computes FOUR independent 8x8x4 products, one per group of 8 lanes: group i = lanes 4i..4i+3 and
// 16+4i..16+4i+3. Within a group, lane j (0..7 in that order) supplies row j of A (its 4 k values in a0.lo, a0.hi,
// a1.lo, a1.hi) and column j of B (same order), and receives 8 outputs:
//   d0 = (rA, cA)  d1 = (rA, cA+1)  d2 = (rB, cA)  d3 = (rB, cA+1)  d4..d7 = the same with columns cA+4, cA+5,
//   where rA = (j & 1) + 4 (j >> 2), rB = rA + 2, cA = 2 ((j >> 1) & 1).
// A Turing 16x8x8 multiply is exactly four 8x8x4 products (two halves of M times two halves of K), so one Volta
// instruction replaces one Turing instruction: group i takes rows 8 (i >> 1).. and k 4 (i & 1)... The partial
// sums of the two K halves live in different groups and are added once, after the K loop (volta_finalize_tile).
#include <cuda_fp16.h>

#include <humming/utils/base.cuh>

#define VOLTA_FULL_MASK 0xffffffffu

// Device code compiled for Volta. NVRTC/nvcc compile device code only here, so the macro is uniform per build.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#define HUMMING_VOLTA 1
#else
#define HUMMING_VOLTA 0
#endif

// Where this lane sits in the m8n8k4 group structure (see the top of this file), and which Turing lanes hold the
// two words of its weight row: (g = j, t = 2kh) and (g = j, t = 2kh + 1).
struct VoltaLane {
  uint32_t group, j, nh, kh, src0, src1;
};

CUDA_INLINE VoltaLane volta_lane() {
  const uint32_t lane = threadIdx.x & 31;
  VoltaLane v;
  v.group = (lane >> 2) & 3;
  v.j = (lane & 3) | ((lane >> 4) << 2);
  v.nh = v.group >> 1;
  v.kh = v.group & 1;
  v.src0 = v.j * 4 + 2 * v.kh;
  v.src1 = v.src0 + 1;
  return v;
}

CUDA_INLINE void volta_ld_shared_v2(uint32_t smem_byte_addr, uint32_t *r) {
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_byte_addr));
}

// ldmatrix.m8n8.x{1,2,4}.b16: lanes 8i..8i+7 give the row addresses of 8x8 matrix i; lane l receives, for each
// matrix i, the two consecutive b16 at row l/4, columns (l%4)*2 and +1. Fetch the row address from the lane that
// holds it and do a plain 32-bit shared load.
template <int count>
CUDA_INLINE void volta_ldmatrix(uint32_t smem_row_addr, uint32_t *a) {
  const uint32_t lane = threadIdx.x & 31;
  PRAGMA_UNROLL
  for (int i = 0; i < count; i++) {
    uint32_t row_addr = __shfl_sync(VOLTA_FULL_MASK, smem_row_addr, i * 8 + (lane >> 2));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(a[i]) : "r"(row_addr + (lane & 3) * 4));
  }
}

// The .trans form: lane l receives, for matrix i, elements (row (l%4)*2, col l/4) and (row (l%4)*2+1, col l/4).
template <int count>
CUDA_INLINE void volta_ldmatrix_trans(uint32_t smem_row_addr, uint32_t *a) {
  const uint32_t lane = threadIdx.x & 31;
  PRAGMA_UNROLL
  for (int i = 0; i < count; i++) {
    uint32_t addr_lo = __shfl_sync(VOLTA_FULL_MASK, smem_row_addr, i * 8 + (lane & 3) * 2);
    uint32_t addr_hi = __shfl_sync(VOLTA_FULL_MASK, smem_row_addr, i * 8 + (lane & 3) * 2 + 1);
    uint16_t lo, hi;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(lo) : "r"(addr_lo + (lane >> 2) * 2));
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(hi) : "r"(addr_hi + (lane >> 2) * 2));
    a[i] = (uint32_t)lo | ((uint32_t)hi << 16);
  }
}

CUDA_INLINE float2 volta_h2f(uint32_t v) {
  __half2 h = *reinterpret_cast<__half2 *>(&v);
  return __half22float2(h);
}

// Turing's m16n8k8 fragment layout (g = lane/4, t = lane%4):
//   a[0] = {A[g][2t], A[g][2t+1]}   a[1] = {A[g+8][2t], A[g+8][2t+1]}      A: 16x8
//   b[0] = {B[2t][g], B[2t+1][g]}                                          B: 8x8 (K x N)
//   d[0..1] = D[g][2t..2t+1]         d[2..3] = D[g+8][2t..2t+1]
// Reference version: gather the full A rows and B columns with shuffles, multiply with fp32 FMAs. Correct and slow.
CUDA_INLINE void volta_mma_m16n8k8_f32(const uint32_t *a, const uint32_t *b, float *d) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t g = lane >> 2, t = lane & 3;
  float a_row0[8], a_row1[8], b_col0[8], b_col1[8];
  PRAGMA_UNROLL
  for (int tt = 0; tt < 4; tt++) {
    float2 v0 = volta_h2f(__shfl_sync(VOLTA_FULL_MASK, a[0], g * 4 + tt));
    float2 v1 = volta_h2f(__shfl_sync(VOLTA_FULL_MASK, a[1], g * 4 + tt));
    a_row0[2 * tt] = v0.x;
    a_row0[2 * tt + 1] = v0.y;
    a_row1[2 * tt] = v1.x;
    a_row1[2 * tt + 1] = v1.y;
    float2 w0 = volta_h2f(__shfl_sync(VOLTA_FULL_MASK, b[0], (2 * t) * 4 + tt));
    float2 w1 = volta_h2f(__shfl_sync(VOLTA_FULL_MASK, b[0], (2 * t + 1) * 4 + tt));
    b_col0[2 * tt] = w0.x;
    b_col0[2 * tt + 1] = w0.y;
    b_col1[2 * tt] = w1.x;
    b_col1[2 * tt + 1] = w1.y;
  }
  float d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3];
  PRAGMA_UNROLL
  for (int k = 0; k < 8; k++) {
    d0 = fmaf(a_row0[k], b_col0[k], d0);
    d1 = fmaf(a_row0[k], b_col1[k], d1);
    d2 = fmaf(a_row1[k], b_col0[k], d2);
    d3 = fmaf(a_row1[k], b_col1[k], d3);
  }
  d[0] = d0;
  d[1] = d1;
  d[2] = d2;
  d[3] = d3;
}

// ---- the fast path: Volta's m8n8k4 --------------------------------------------------------------------------------

// The lane that is index j of group i (see the top of this file).
CUDA_INLINE uint32_t volta_lane_of(uint32_t group, uint32_t j) {
  return j < 4 ? 4 * group + j : 16 + 4 * group + (j - 4);
}

CUDA_INLINE void volta_mma_884(const uint32_t *a, const uint32_t *b, float *c) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}

// One Turing m16n8k8 step with Turing-layout a and b, accumulated in Volta's layout: this lane's 8 outputs of its
// group's 8x8x4 product, in c_lo[0..3] (d0..d3) and c_hi[0..3] (d4..d7). Six shuffles move the operands into the
// group layout; the tensor core does the rest. The accumulators are not in Turing layout until volta_finalize_tile.
CUDA_INLINE void volta_mma_m16n8k8_fast(const uint32_t *a, const uint32_t *b, float *c_lo, float *c_hi) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t group = (lane >> 2) & 3;
  const uint32_t j = (lane & 3) | ((lane >> 4) << 2);
  const uint32_t mh = group >> 1, kh = group & 1;
  // Turing lanes (g = j, t = 2kh) and (g = j, t = 2kh + 1) hold A[8mh + j][4kh..4kh+3] and B[4kh..4kh+3][j].
  const uint32_t src0 = j * 4 + 2 * kh, src1 = src0 + 1;
  uint32_t a0_0 = __shfl_sync(VOLTA_FULL_MASK, a[0], src0);
  uint32_t a1_0 = __shfl_sync(VOLTA_FULL_MASK, a[1], src0);
  uint32_t a0_1 = __shfl_sync(VOLTA_FULL_MASK, a[0], src1);
  uint32_t a1_1 = __shfl_sync(VOLTA_FULL_MASK, a[1], src1);
  uint32_t av[2] = {mh ? a1_0 : a0_0, mh ? a1_1 : a0_1};
  uint32_t bv[2] = {__shfl_sync(VOLTA_FULL_MASK, b[0], src0), __shfl_sync(VOLTA_FULL_MASK, b[0], src1)};
  float c[8] = {c_lo[0], c_lo[1], c_lo[2], c_lo[3], c_hi[0], c_hi[1], c_hi[2], c_hi[3]};
  volta_mma_884(av, bv, c);
  c_lo[0] = c[0];
  c_lo[1] = c[1];
  c_lo[2] = c[2];
  c_lo[3] = c[3];
  c_hi[0] = c[4];
  c_hi[1] = c[5];
  c_hi[2] = c[6];
  c_hi[3] = c[7];
}

// One m8n8k4 on operands already in the group layout, accumulating into the two four-register halves.
CUDA_INLINE void volta_mma_884_split(const uint32_t *a, const uint32_t *b, float *c_lo, float *c_hi) {
  float c[8] = {c_lo[0], c_lo[1], c_lo[2], c_lo[3], c_hi[0], c_hi[1], c_hi[2], c_hi[3]};
  volta_mma_884(a, b, c);
  c_lo[0] = c[0];
  c_lo[1] = c[1];
  c_lo[2] = c[2];
  c_lo[3] = c[3];
  c_hi[0] = c[4];
  c_hi[1] = c[5];
  c_hi[2] = c[6];
  c_hi[3] = c[7];
}

// After the K loop: add the two K halves (held by partner groups) and put the 16x8 tile into Turing's layout in
// c_lo[0..3]. 24 shuffles per tile, once per tile.
CUDA_INLINE void volta_finalize_tile(float *c_lo, float *c_hi) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t group = (lane >> 2) & 3;
  const uint32_t j = (lane & 3) | ((lane >> 4) << 2);
  const uint32_t partner = volta_lane_of(group ^ 1, j); // same rows and columns, the other K half
  float sum[8] = {c_lo[0], c_lo[1], c_lo[2], c_lo[3], c_hi[0], c_hi[1], c_hi[2], c_hi[3]};
  PRAGMA_UNROLL
  for (int q = 0; q < 8; q++)
    sum[q] += __shfl_sync(VOLTA_FULL_MASK, sum[q], partner);
  // This lane, as Turing lane (g, t), needs D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1]. All four sit in the
  // lane with group index js of the K-half-0 group of the matching M half, in registers r0 and r0 + 1.
  const uint32_t g = lane >> 2, t = lane & 3;
  const uint32_t js = (g & 1) | ((t & 1) << 1) | ((g >> 2) << 2);
  const uint32_t r0 = ((t >> 1) << 2) | (((g >> 1) & 1) << 1);
  float out[4];
  PRAGMA_UNROLL
  for (uint32_t mh = 0; mh < 2; mh++) {
    const uint32_t src = volta_lane_of(mh * 2, js);
    PRAGMA_UNROLL
    for (uint32_t rb = 0; rb < 4; rb++) {
      PRAGMA_UNROLL
      for (uint32_t b = 0; b < 2; b++) {
        float v = __shfl_sync(VOLTA_FULL_MASK, sum[rb * 2 + b], src);
        if (r0 == rb * 2) out[mh * 2 + b] = v;
      }
    }
  }
  c_lo[0] = out[0];
  c_lo[1] = out[1];
  c_lo[2] = out[2];
  c_lo[3] = out[3];
  c_hi[0] = 0.f;
  c_hi[1] = 0.f;
  c_hi[2] = 0.f;
  c_hi[3] = 0.f;
}

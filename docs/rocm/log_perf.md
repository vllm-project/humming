# Dense weight-only MFMA kernel: large-M performance log (gfx950 / MI355X)

Kernel: `humming/include/humming/kernel/humming_mfma.cuh`, tile heuristic
`humming/tune/gfx950.py`. Goal: close the large-M gap to hipBLASLt fp16 without
regressing M <= 256.

Method (unless noted): `python -m tests.rocm.bench_mfma --ms ... --shapes NxK --dtype ...`
on one GPU of a shared node, best of 2 runs, HIP-graph timing, weights rotated over
>= 1 GB. A fp16/bf16, uint4 symmetric, group 128. "hipBLASLt" = `torch.matmul` on the
dequantized weight (same dtype). Tile notation `wm,wn,tm,tn,stages` = waves (m, n),
16x16 MFMA tiles per wave (m, n), pipeline stages; block tile = (wm*tm*16) x (wn*tn*16).
Starting point: `4288c98` (default tile for large M: 1,4,8,2,2 lds = 128x128, 2 blocks/CU).

## Iteration 0: where the time goes (profile)

8192x28672 (N x K), M=4096, fp16: 1849 us, hipBLASLt 1283 us (0.69x).
MFMA peak for this GEMM (2*M*N*K = 1.92 TFLOP at 2516 TF) is 765 us, so hipBLASLt runs at
60% of MFMA peak and Humming at 41%.

Main-loop profile (rocprofv3 counters + ISA instruction mix, `/tmp/perfwork/loops.py`),
fp16, measured right after iteration 1:

| kernel / tile | LDS reads / MFMA | VALU / MFMA |
|---|---|---|
| Humming 8x2 (128x128) | 0.5 | 1.96 |
| Humming 8x4 (128x256) | 0.25 | 1.95 |
| hipBLASLt MT256x256 | 0.25 | 0.04 |

The dequant is 15 VALU per 8-value int4 fragment (3 lshr, 4 bitop3, 4 pk_add, 4 pk_mul).

Co-issue microbenchmark (2 waves/SIMD, a dependent-free MFMA stream with N independent
VALU interleaved): VALU per MFMA 2 -> +14% time, 3 -> +50%, 4 -> +53%. MFMA 16x16x32 is
16 cycles and the VALU in the same SIMD only hides behind it up to about 2 ops per
MFMA. So the dequant VALU is on the critical path at ~1.95/MFMA and the first lever
is VALU per MFMA, the second is reuse of each dequantized B fragment over more M rows
(bigger tm or wm).

hipBLASLt for the same GEMM: MT256x256 macro tile, 4 waves with 128x128 wave tiles,
135 KB LDS, 256 persistent WGs, 1 wave/SIMD, 59.5% MFMA utilization.

## Iteration 1: direct-to-LDS A loads (`68df7fd`, kept)

Hypothesis: A staging via registers (`global_load` -> `ds_write`) costs VGPRs and VALU
issue slots; `buffer_load_dwordx4 ... lds` writes LDS directly and frees both, which in
turn keeps the 8x4 (128x256) main loop spill-free (35 VGPRs still spill in the epilogue).

| fp16 8192x28672 M=4096 | before | after |
|---|---|---|
| 8x2 (128x128) | 1849 | 1698 |
| 8x4 (128x256) | spills | 1476 |

Removing the `vmcnt` wait on the A chunk (relying only on the barrier) changed nothing.

Side effect found later: with 2 stages, the DMA path must drain `vmcnt(0)` before issuing
the next chunk (the LDS slot being overwritten is the one just read), which exposes full
load latency at small M. Fixed in iteration 5.

## Iteration 2: shift-light integer dequant (`57ac1de`, kept)

Hypothesis: the dequant does `(w >> shift) & mask | magic` per field, i.e. one shift per
nibble. A field at bit offset r inside a 16-bit half can be decoded in place:
`(w & (mask << r)) | magic(2^(10-r))` is the fp16 number `2^(10-r) + code*2^0` scaled
so that subtracting `2^(10-r) + zp` (one `v_pk_add_f16` / `v_pk_fma_f16`) gives the exact
code. One shift per 32-bit word instead of per field. Bit-identical output (all values
are small integers, exact in fp16).

| fp16 int4 | before | after |
|---|---|---|
| VALU per fragment (8 values) | 15 | 13 |
| main-loop VALU / MFMA (lshr count 96 -> 32) | 1.95 | 1.66 |

| fp16 8192x28672 M=4096 | before | after |
|---|---|---|
| 8x2 | 1698 | 1626 |
| 8x4 | 1476 | 1472 |

Main loop fp16 is now 1.66 VALU/MFMA. 8x4 barely moves (it already amortizes the
dequant over 8 M-tiles), 8x2 gains 4%.

## Iteration 3: bf16 dequant via `v_fma_mix_f32` (`cb9476d`, kept)

bf16 has no 2^(10-r) trick on the packed bf16 side (7-bit mantissa), so the old path was
int -> f32 -> scale -> bf16, 25 VALU/fragment. New path: decode the field as fp16 with
the shift-light trick, then `v_fma_mix_f32(x_f16, scale_f32, -(2^(10-r)+zp)*scale)`
(fp16 operand converted in the FMA), then `v_cvt_pk_bf16_f32`. The FMA is exact (all
products are <= 19 significant bits), so the result is rounded once -> bit-identical to
the reference `((code - zp) * scale).to(bf16)`. LLVM never formed `fma_mix` from the C++,
so it is inline asm with a host fallback. Group scale is folded into the dequant
(`run_scaled`) for bf16.

| VALU per int4 fragment | before | after |
|---|---|---|
| bf16 | 25 | 17 |

| bf16 8192x28672 M=4096 | before | after |
|---|---|---|
| 8x2 | 1909 | 1677 |
| 8x4 | 2198 (spills) | 1779 (spills) |

Main-loop VALU/MFMA for bf16 8x2: 2.2.

Rejected variant: subtract in fp16 then mix (`HUMMING_BF16_SUBF16`) -> 64 VGPR spills in
the 8x4 tile, 3113 us.

bf16 still cannot use 8x4: the per-residual f32 offsets push it to 56 spilled VGPRs
(1779 us vs 1677 us for 8x2). `amdgpu-schedule-*` / `-mllvm` scheduler flags did not remove
the spills. Other bf16 tiles at M=4096, 8192x28672 (us): 8x3 1743, 2x2 waves 8x4 2004,
2x4 waves 8x2 1810, 16x2 1836. So bf16 stays on 8x2.

## Iteration 4: 128x256 tile for large-M fp16 (`1112a1d`, kept)

Hypothesis: 8x4 halves the A LDS reads per MFMA and dequant per MFMA stays the same, but
it has half as many blocks, so it only wins when the grid fills the GPU. Swept tile x split
over N in {4096, 8192, 28672, 57344} and M in 64..4096 (`/tmp/perfwork/sweep_1422.txt`).
8x4 wins once `ceil(N/256) * ceil(M/128) >= 1.5 * CUs`. Boundary samples (8x2 vs 8x4, us):

| N | M | 8x2 | 8x4 |
|---|---|---|---|
| 57344 | 128 | 128.8 | 126.3 |
| 57344 | 192 | 231 | 207 |
| 28672 | 448 | 119 | 106 |
| 8192 | 1536 | 637 | 647 |
| 4096 | 3072 | 98.2 | 101.4 |

Change: `wide_tile_min_m(N, cus) = max(64, 128 * (ceil(1.5*cus / ceil(N/256)) - 1))`;
`select_tile` uses 8x4 for fp16 above it, and `get_m_intervals` splits the last M bucket
at that N-dependent bound so the compiled config table matches. bf16 is excluded (spills).

| fp16 8192x28672 M=4096 | before (8x2) | after (8x4) |
|---|---|---|
| us | 1626 | 1478 |

The end-to-end effect on the other shapes is in the final table (M=256..4096 for
57344x8192 and M=4096 for 4096x4096 switch to 8x4).

Tests: `test_wide_tile` (bits 1-5, 8; g0/g128; zp; M 77/300; fp16+bf16) and
`test_tile_buckets` (interval consistency, tuner == test mirror).

## Iteration 5: 3-stage pipeline for the M <= 64 tiles (`3682ec7`, kept)

Bisect showed iteration 1 regressed small M (fp16 8192x28672 M=16: 39.2 -> 44.9 us,
M=64: 51.2 -> 54). Cause: the DMA path with 2 stages waits `vmcnt(0)` before the next
chunk is issued. With 3 stages there is always one chunk in flight. The M <= 32 (2x2) and
M <= 64 (4x2) LDS tiles now use 3 stages:

| fp16 8192x28672 | 4288c98 | 68df7fd | 3682ec7 |
|---|---|---|---|
| M=16 | 38.6 | 44.9 | 35.6 |
| M=64 | 51.2 | 54.0 | 47.3 |

3 stages for 8x2 at M=128/256 was mixed or worse (kept at 2).

## Iteration 6: wide tile only for <= 4-bit weights (`b464ea9`, kept)

The first full `bench_baselines.py` run after iteration 4 showed fp16 W8 at large M 1.6-2.5x
slower than before (e.g. llama3-70b/down M=4096 1953 -> 3515 us; W8 fp16 geomean vs
hipBLASLt at M=4096 0.35x, while bf16 W8 on 8x2 was 0.69x). The sweep in iteration 4 was
W4 only. Per bit width, fp16 8192x28672 (us, 8x2 vs 8x4):

| bits | M=1024 8x2 | M=1024 8x4 | M=4096 8x2 | M=4096 8x4 |
|---|---|---|---|---|
| 1 | 406.7 | 374.6 | 1529.9 | 1360.6 |
| 2 | 415.4 | 399.8 | 1593.8 | 1414.1 |
| 3 | 434.9 | 428.4 | 1668.3 | 1504.3 |
| 5 | 466.8 | 1269.2 | 1771.5 | 3410.1 |
| 6 | 471.5 | 965.2 | 1781.5 | 2591.4 |
| 7 | 479.2 | 1221.5 | 1834.9 | 3957.4 |
| 8 | 458.6 | 1253.4 | 1779.7 | 3521.0 |

For 5-8 bits a B fragment is two packed words per lane instead of one, on top of the 128 fp32
accumulators per lane of the 8x4 tile. Compiler resource remarks for fp16 8x4: VGPR spills
35 (W4, epilogue only, main loop spill-free) / 61 (W5) / 78 (W8); scratch 136 / 208 / 236 B
per lane. `wide_tile_allowed(is_bfloat16,
num_bits)` now gates the 8x4 tile on fp16 and `num_bits <= 4`; the tuner reads
`layer_config.b_dtype.num_bits`. Tests: `test_tile_buckets` over bits 1/4/5/8 and
`test_heuristics_wide_tile_by_dtype` (LayerConfig -> `get_configs` / `get_config`).

## Iteration 7: 32x128 tile for K-starved M <= 64 (`64441f3`, kept)

The same baseline run showed fp16 W8 zp M=64 on small shapes 3.7-3.9% slower than `4288c98`
(llama3-8b o_proj / qkv). Bisect (fixed tile 1,4,4,2,2, best of 3, us):

| commit | 4096x4096 M=16 | 4096x4096 M=64 | 5120x5120 M=16 | 5120x5120 M=64 |
|---|---|---|---|---|
| 4288c98 | 12.8 | 13.4 | 16.2 | 17.5 |
| 68df7fd (DMA A) | 13.3 | 13.8 | 17.2 | 18.3 |
| 57ac1de / HEAD | 13.3 | 13.8 | 17.2 | 18.3 |

So the direct-to-LDS A load costs W8 3-6% on these grids (2 vs 3 stages made no difference).
The register-staged A path no longer exists, so instead looked at the tile: these are the
shapes where the 64x128 tile hits the split-K cap (8) with only 4-5 chunks of 128 per split.
Sweep at M=48/64, 64x128 vs 32x128 (ratio > 1 = 32x128 faster), fp16:

| N x K | W4 M=64 | W8 M=64 | 64x128 chunks/split |
|---|---|---|---|
| 4096x4096 | 1.12 | 1.05 | 4 |
| 6144x4096 | 1.17 | 1.09 | 4 |
| 5120x5120 | 1.14 | 1.08 | 5 |
| 4096x14336 | 1.01 | 0.94 | 14 |
| 8192x8192 | 0.97 | 0.92 | 16 |
| 28672x4096 | 0.89 | 0.80 | 16 |
| 8192x28672 | 0.89 | 0.85 | 56 |
| 57344x8192 | 0.78 | 0.65 | 64 |

bf16 follows the same pattern. Rule: for M in (32, 64] use 32x128 when the 64x128 grid has
< 8 chunks per split (`is_k_starved`; one block row for every M in the bucket, so the tile
is M-independent and the compiled config table is unchanged). Default heuristic vs `4288c98`
(best of 2, us):

| dtype | bits | M | 4096x4096 | 6144x4096 | 5120x5120 | 4096x14336 |
|---|---|---|---|---|---|---|
| fp16 | 4 | 64 | 12.9 -> 10.6 | 14.7 -> 12.4 | 15.7 -> 13.8 | 20.4 -> 19.0 |
| fp16 | 8 | 64 | 13.5 -> 11.3 | 16.7 -> 14.8 | 17.7 -> 16.7 | 24.0 -> 22.7 |
| bf16 | 4 | 64 | 13.8 -> 11.4 | 16.0 -> 13.5 | 17.6 -> 15.2 | 23.3 -> 20.3 |
| bf16 | 8 | 64 | 14.2 -> 11.9 | 17.6 -> 15.4 | 18.2 -> 17.2 | 24.5 -> 23.4 |

Remaining after this step: W8 fp16 at M=16 on small K was 1-3% slower than `4288c98`
(5120x5120 12.8 -> 13.2 us), the residual cost of the DMA A path; fixed by iteration 8.
Test: `test_m64_tile_k_starved` (tuner and mirror for 6 shapes, M 33/48/64);
`test_tile_buckets` now also runs K = 4096 and 14336.

## Iteration 8: 16-row tile for M <= 16, 32x64 tile for small-N M <= 32 (`e86ea0d`, kept)

Looking for the last W8 M=16 points, a tile sweep at M=16/32 (all LDS + scale_b, 3 stages,
best of 2, us) showed M in (8, 16] was running 32-row blocks with half the rows empty:

| fp16 W4 M=16 | 2x2 (32x128, old) | 1x2 (16x128) | 2x1 (32x64) | 1x1 (16x64) |
|---|---|---|---|---|
| 4096x4096 | 10.3 | 9.1 | 9.3 | 9.1 |
| 5120x5120 | 11.5 | 10.0 | 10.2 | 9.6 |
| 4096x14336 | 15.5 | 13.9 | 14.0 | 13.2 |
| 8192x8192 | 14.8 | 13.5 | 13.7 | 12.7 |
| 28672x4096 | 19.0 | 16.9 | 19.0 | 16.6 |
| 8192x28672 | 36.9 | 32.8 | 34.9 | 30.9 |
| 57344x8192 | 55.9 | 50.9 | 59.3 | 53.7 |

1x2 is never slower than 2x2 (fp16/bf16, W4/W8) and within ~6% of the best everywhere;
1x1 is better on some shapes but loses at large N and on W8 8192x8192 (19.1 vs 17.9), so
1x2 is used for the whole bucket.

| M=32 | 2x2 (old) | 2x1 (32x64) |
|---|---|---|
| fp16 W4 4096x4096 | 10.5 | 9.4 |
| fp16 W4 6144x4096 | 11.3 | 10.2 |
| fp16 W4 4096x14336 | 15.7 | 14.9 |
| fp16 W4 8192x8192 | 15.1 | 14.4 |
| fp16 W8 8192x8192 | 18.8 | 20.2 |
| fp16 W4 28672x4096 | 19.5 | 21.0 |
| fp16 W4 57344x8192 | 59.8 | 67.6 |

Rule for (16, 32]: 2x1 when `N // 128 < num_cus // 4` (N <= 6144 on 256 CUs), else 2x2 (the
same form as the decode rule). This adds one M bucket: (0,8], (8,16], (16,32], (32,64], then
one or two large-M buckets, so a layer compiles at most six kernels.

Default heuristic vs `4288c98` over the 8 shapes above, best of 2:

| dtype | bits | M=12 | M=16 | M=24 | M=32 | slowest point |
|---|---|---|---|---|---|---|
| fp16 | 4 | 1.19x | 1.19x | 1.15x | 1.14x | 1.06x |
| fp16 | 8 | 1.07x | 1.07x | 1.08x | 1.08x | 1.00x |
| bf16 | 4 | 1.28x | 1.28x | 1.22x | 1.22x | 1.12x |
| bf16 | 8 | 1.13x | 1.13x | 1.14x | 1.13x | 1.03x |

(geomean speedup; the slowest point for fp16 W8 is 57344x8192 at 0.998-1.009x.)
Tests: `test_small_m_tiles` (tuner + mirror), new tiles in `test_tiles_and_splits`, and
`test_small_shapes` now includes M=32 (all bits / groups / zp, exercises 2x1).

## Iteration 9: 32x64 tile only for N <= 4096 or K <= N (`1d082f1`, kept)

The baseline grid at `e86ea0d` showed fp16 W8 qwen3-32b/down (5120x25600) at M=32 6% slower
than before. A/B vs `4288c98` (best of 3) confirmed it: 37.1 -> 39.8 us (W8), 37.6 -> 40.4 (W8
zp); the other >3% grid flags (M=1 decode, M=64) were noise (within +-2% in the A/B). Sweep
of 2x2 vs 2x1 at M=32 (us, best of 2, 2x2 / 2x1 speedup):

| N x K | W4 2x2 | W4 2x1 | W8 2x2 | W8 2x1 |
|---|---|---|---|---|
| 4096x4096 | 9.7 | 8.4 (1.15x) | 10.5 | 9.2 (1.14x) |
| 4096x8192 | 13.1 | 10.9 (1.20x) | 15.2 | 13.7 (1.11x) |
| 4096x14336 | 18.3 | 15.5 (1.18x) | 23.7 | 18.9 (1.25x) |
| 4096x28672 | 30.5 | 25.7 (1.19x) | 40.9 | 32.4 (1.26x) |
| 5120x5120 | 10.9 | 10.2 (1.07x) | 11.9 | 12.0 (0.99x) |
| 6144x4096 | 10.2 | 9.1 (1.12x) | 12.0 | 10.9 (1.10x) |
| 6144x8192 | 13.9 | 13.0 (1.07x) | 17.3 | 18.2 (0.95x) |
| 5120x14336 | 19.2 | 18.9 (1.02x) | 25.4 | 25.0 (1.02x) |
| 6144x12288 | 17.6 | 17.3 (1.02x) | 22.9 | 23.9 (0.96x) |
| 6144x16384 | 21.4 | 22.3 (0.96x) | 28.3 | 29.9 (0.95x) |
| 5120x25600 | 28.6 | 29.0 (0.99x) | 38.1 | 39.6 (0.96x) |

A K-chunks-per-split threshold does not separate these (4096x28672 wins with 28 chunks per
split, 6144x16384 loses with 16); N does: N = 4096 always wins, N = 5120/6144 only with K <= N.
New rule: 2x1 when `N // 128 <= num_cus // 8`, or `N // 128 < num_cus // 4` and `K <= N`.
After: 5120x25600 M=32 W8 37.2 us (base 37.1), W4 zp 28.1 (base 29.7). M = 16/24/32 over the 12
grid shapes vs `4288c98` (geomean / min): fp16 W4 1.21/1.10/1.10 (min 1.02), fp16 W8
1.07/1.03/1.03 (min 0.97, 10240x8192 M=24 25.3 -> 26.0, the 3-stage 2x2 tile), bf16 W4
1.32/1.19/1.19, bf16 W8 1.13/1.09/1.09.
`test_small_m_tiles` covers 8 (N, K) pairs including both sides of the K <= N rule.

## Rejected experiments

| experiment | fp16 8192x28672 M=4096 | verdict |
|---|---|---|
| STREAM_A reorder (A fragment reads interleaved with B dequant) | 1509-1520 vs 1476 | worse |
| `sched_group_barrier` MFMA/VALU interleave, 8x2 | 1637-1855 vs 1626 | worse, main-loop spills |
| same, 8x4 | 2668-3743 | much worse, spills |
| same, 16x4 | 1927-2330 | worse |
| 16x4 tile (occupancy 1) | 1698 | worse: accumulator in AGPRs is rotated, ~416 `v_accvgpr` moves per 512 MFMA |
| 16x2 / 12x4 / 16x3 | 1815 / 1710 / 1762 | worse |
| rowsum zero-point correction (emulated: drop the per-fragment subtract, VALU 1.66 -> 1.22 / MFMA) | 1434 vs 1477 | only 3%, needs an extra rowsum(A) pass and an epilogue; not implemented |

### 32x32x16 MFMA layout: rejected

Considered switching to `v_mfma_f32_32x32x16_f16`. It does not reduce LDS reads or VALU
per FLOP for this kernel: the dequant cost is per B element, and a 32x32 tile consumes
the same B elements per FLOP as a 16x16 wave tile with the same M extent (a 128-row wave
tile already reuses each dequantized B fragment 8 times). A-fragment LDS traffic per FLOP
is set by the wave tile shape, not by the MFMA shape. It would also need a new B
pre-shuffle layout (32-lane K-groups), new dequant indexing and new tests for every bit
width, so it is not worth doing.

## Bound: why M=1024 >= 1.1x and M=4096 >= 0.9x are not reached

Both M=1024 and M=4096 are compute-bound for both kernels (weight bytes are < 10% of the
time at the measured TB/s).

* Fake-dequant run (B registers fed without any dequant, same tile 8x4): 1243 us at
  8192x28672 M=4096, vs hipBLASLt 1283 us (1.03x). With no dequant at all the kernel is
  only 3% faster than hipBLASLt; 12x4 / 16x4 without dequant give 1516 / 1419 us, so a
  bigger tile does not help either (registers / AGPR rotation). The real kernel is at
  1451-1478 us; the remaining 17-19% over fake-dequant is the dequant VALU (1.66/MFMA, above the
  ~1-2 VALU free co-issue window from the microbenchmark). The rowsum emulation shows the
  dequant could only shrink by about 3% more without changing the format.
* M=4096 target 0.9x = 1426 us at 8192x28672: reached for 57344x8192 at 0.89x and
  8192x28672 at 0.87x (2-3% short). 4096x4096 is 0.82x (115-117 vs 95-96 us, 8x4, 512 blocks =
  2 per CU): with K=4096 each block runs only 32 chunks, so the fixed per-block cost
  (pipeline fill, the 128x256 epilogue with its 35 spilled VGPRs) weighs more; not profiled
  further.
* M=1024 at 1.1x would need 331 us at 8192x28672, i.e. 57.8% of MFMA peak, while
  hipBLASLt itself runs at 52.5% (364 us) with no dequant. At M=1024, N=8192 there are
  only 32x8 = 256 128x256 blocks (1/CU), so the 8x2 tile (512 blocks) is used and is
  at 45%. Getting there would need a kernel faster than hipBLASLt on a plain fp16 GEMM.
  4096x4096 M=1024 reaches 1.06x (bf16 1.11x); 57344x8192 0.97x.
* bf16 vs fp16 (target <= 5%): 14% slower at M=4096 (1661 vs 1451 us) because bf16 is on
  8x2 (17 vs 13 VALU/fragment pushes 8x4 into spills). At M <= 256 bf16 is 3-9% slower, except 57344x8192 M=256 at 13% (245 vs 216 us).

## Results: before (4288c98) -> after (e86ea0d; `1d082f1` selects the same tiles for these points)

us, best of 2; ratio = hipBLASLt us / Humming us (>1 = Humming faster).

| dtype | N x K | M | before | after | speedup | hipBLASLt | before ratio | after ratio |
|---|---|---|---|---|---|---|---|---|
| fp16 | 4096x4096 | 1 | 7.2 | 7.2 | 1.00x | 12.1 | 1.68x | 1.68x |
| fp16 | 4096x4096 | 16 | 10.2 | 9.2 | 1.11x | 10.1 | 0.99x | 1.10x |
| fp16 | 4096x4096 | 64 | 12.9 | 10.6 | 1.22x | 13.0 | 1.01x | 1.23x |
| fp16 | 4096x4096 | 256 | 22.1 | 20.7 | 1.07x | 22.5 | 1.02x | 1.09x |
| fp16 | 4096x4096 | 1024 | 45.6 | 39.7 | 1.15x | 42.1 | 0.92x | 1.06x |
| fp16 | 4096x4096 | 4096 | 143.8 | 115.1 | 1.25x | 94.7 | 0.66x | 0.82x |
| fp16 | 8192x28672 | 1 | 34.5 | 34.2 | 1.01x | 75.8 | 2.20x | 2.22x |
| fp16 | 8192x28672 | 16 | 38.6 | 32.5 | 1.19x | 76.9 | 1.99x | 2.37x |
| fp16 | 8192x28672 | 64 | 51.2 | 47.1 | 1.09x | 81.3 | 1.59x | 1.73x |
| fp16 | 8192x28672 | 256 | 150.4 | 125.3 | 1.20x | 122.3 | 0.81x | 0.98x |
| fp16 | 8192x28672 | 1024 | 482.6 | 421.6 | 1.14x | 358.8 | 0.74x | 0.85x |
| fp16 | 8192x28672 | 4096 | 1841.7 | 1451.2 | 1.27x | 1262.5 | 0.69x | 0.87x |
| fp16 | 57344x8192 | 1 | 49.9 | 49.6 | 1.01x | 155.1 | 3.11x | 3.13x |
| fp16 | 57344x8192 | 16 | 60.4 | 50.3 | 1.20x | 161.2 | 2.67x | 3.20x |
| fp16 | 57344x8192 | 64 | 88.3 | 80.1 | 1.10x | 159.3 | 1.80x | 1.99x |
| fp16 | 57344x8192 | 256 | 268.4 | 215.9 | 1.24x | 212.7 | 0.79x | 0.99x |
| fp16 | 57344x8192 | 1024 | 960.0 | 758.9 | 1.26x | 734.2 | 0.76x | 0.97x |
| fp16 | 57344x8192 | 4096 | 3765.3 | 2946.6 | 1.28x | 2631.9 | 0.70x | 0.89x |
| bf16 | 4096x4096 | 1 | 7.7 | 7.7 | 1.00x | 12.2 | 1.58x | 1.58x |
| bf16 | 4096x4096 | 16 | 11.0 | 9.5 | 1.16x | 10.1 | 0.92x | 1.06x |
| bf16 | 4096x4096 | 64 | 13.8 | 11.5 | 1.20x | 17.5 | 1.27x | 1.52x |
| bf16 | 4096x4096 | 256 | 23.9 | 21.7 | 1.10x | 19.6 | 0.82x | 0.90x |
| bf16 | 4096x4096 | 1024 | 50.6 | 40.6 | 1.25x | 45.0 | 0.89x | 1.11x |
| bf16 | 4096x4096 | 4096 | 155.2 | 128.4 | 1.21x | 94.1 | 0.61x | 0.73x |
| bf16 | 8192x28672 | 1 | 36.7 | 36.4 | 1.01x | 76.1 | 2.07x | 2.09x |
| bf16 | 8192x28672 | 16 | 45.2 | 35.4 | 1.28x | 77.2 | 1.71x | 2.18x |
| bf16 | 8192x28672 | 64 | 58.4 | 49.6 | 1.18x | 80.9 | 1.39x | 1.63x |
| bf16 | 8192x28672 | 256 | 165.0 | 129.1 | 1.28x | 119.7 | 0.73x | 0.93x |
| bf16 | 8192x28672 | 1024 | 532.8 | 430.4 | 1.24x | 349.2 | 0.66x | 0.81x |
| bf16 | 8192x28672 | 4096 | 2070.9 | 1661.2 | 1.25x | 1217.1 | 0.59x | 0.73x |
| bf16 | 57344x8192 | 1 | 57.9 | 53.3 | 1.09x | 155.7 | 2.69x | 2.92x |
| bf16 | 57344x8192 | 16 | 75.8 | 54.1 | 1.40x | 158.7 | 2.09x | 2.93x |
| bf16 | 57344x8192 | 64 | 102.1 | 84.0 | 1.22x | 157.2 | 1.54x | 1.87x |
| bf16 | 57344x8192 | 256 | 297.3 | 245.0 | 1.21x | 204.9 | 0.69x | 0.84x |
| bf16 | 57344x8192 | 1024 | 1080.5 | 870.8 | 1.24x | 718.3 | 0.66x | 0.82x |
| bf16 | 57344x8192 | 4096 | 4303.8 | 3412.4 | 1.26x | 2543.7 | 0.59x | 0.75x |

No M <= 256 point is slower than before (the lowest speedup is 1.00x at M=1).
Full baseline grid (W4 sym/zp g128, W8; fp16/bf16; all baseline impls):
`benchmarks/rocm/results/humming_perf_2026-10-28.{csv,md}`.

Grid at `1d082f1` (humming rows, geomean over the 10 shapes and sym/zp; speedup vs the earlier
`humming_vs_baselines_2026-10-28.csv`, ratio = hipBLASLt / Humming):

| | M=1 | 4 | 16 | 32 | 64 | 128 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| fp16 W4 speedup | 1.00 | 1.00 | 1.21 | 1.12 | 1.11 | 1.08 | 1.12 | 1.14 | 1.22 |
| fp16 W4 vs hipBLASLt | 2.01 | 1.89 | 2.00 | 1.95 | 1.54 | 1.11 | 1.05 | 0.94 | 0.86 |
| fp16 W8 speedup | 0.99 | 1.00 | 1.06 | 1.06 | 1.04 | 1.07 | 1.08 | 1.09 | 1.10 |
| fp16 W8 vs hipBLASLt | 1.50 | 1.45 | 1.45 | 1.51 | 1.29 | 0.98 | 0.93 | 0.83 | 0.73 |
| bf16 W4 speedup | 1.03 | 1.03 | 1.31 | 1.21 | 1.18 | 1.13 | 1.18 | 1.22 | 1.22 |
| bf16 W4 vs hipBLASLt | 1.93 | 1.82 | 1.87 | 1.84 | 1.51 | 1.10 | 0.98 | 0.88 | 0.75 |
| bf16 W8 speedup | 1.01 | 1.00 | 1.13 | 1.13 | 1.11 | 1.10 | 1.15 | 1.20 | 1.20 |
| bf16 W8 vs hipBLASLt | 1.53 | 1.45 | 1.43 | 1.51 | 1.31 | 0.98 | 0.88 | 0.80 | 0.69 |

The grid flags 3 M=1 points as 4-10% slower than the earlier file (llama3-70b/down W4,
qwen3-32b/gate_up W8 zp). Rerunning the same harness for base `4288c98` and HEAD in one
session gives equal times (fp16 W4 zp llama3-70b/down 38.2 vs 38.0 us; W8 zp qwen3 gate_up
51.5-55.3 vs 55.2 us). The M=1 decode tile and its code path are unchanged, so these are
differences between runs, not regressions.

Tests: `pytest tests/rocm` 709 passed / 40 skipped at start -> 1111 passed / 84 skipped.

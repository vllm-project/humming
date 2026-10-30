# Low-bit activation GEMM log (gfx950 / MI355X)

Kernel: `humming/include/humming/kernel/humming_mfma_f8.cuh` (new file; the FP16/BF16 kernel
`humming_mfma.cuh` is untouched). Planning / tiles: `humming/kernel/humming_mfma_f8.py`, weight
repack: `humming/rocm/layout_f8.py`, launcher: `csrc/launcher/mfma_f8.h`. Tests:
`tests/rocm/test_layer_f8.py` (294), `tests/rocm/test_mfma_f8_layout.py` (76). Benchmark:
`benchmarks/rocm/bench_baselines.py` (entries `humming_w8a8_*`, `humming_w4a8_*`,
`humming_w4a4_mxfp4`), results `benchmarks/rocm/results/humming_f8_2026-10-28.{csv,md}`.

All numbers: MI355X GPU 4, bf16 output, kernels captured in a HIP graph, best of N (the node is
shared; +-5 % noise remains). "GEMM-only" timings call `layer(q, input_scale=s)` with
`q, s = may_quant_input(layer.humming_config, x)`; `_e2e` entries include the activation
quantization kernel.

## Support matrix

Same `LayerConfig` / `HummingLayer` schema as CUDA; anything else raises `NotImplementedError`
("Humming on ROCm ... does not support ...") from `plan_mfma_f8` at layer build / forward.

| activation | weight | how B reaches the MFMA | weight scales |
|---|---|---|---|
| FP8 e4m3 / e5m2 | FP8 e4m3 / e5m2, FP6 e2m3 / e3m2, FP4 e2m1 | native operand of `v_mfma_scale_f32_16x16x128_f8f6f4` | tensor, channel, group (FP32 / BF16), block 128x128, MX E8M0 g32 (in hardware) |
| FP8 | INT1-5 symmetric, UINT1-4 + zero point, small FP formats | `v_perm` byte LUT to e4m3 (W4A8) | channel, group 32..256+, tensor |
| INT8 | INT1-8 (zero point up to 7 bits) | `v_mfma_i32_16x16x64_i8`, exact INT32 partials flushed per scale group | channel, group |
| INT8 | MXFP4 | LUT to doubled INT8 (x0.5 folded into the epilogue) | E8M0 g32 |
| MXFP4 (e2m1, E8M0 g32) | MXFP4 / MXFP6 / MXFP8 | native mixed f8f6f4 formats, both E8M0 scales in hardware | MX |
| MXFP4 | INT / small FP | LUT to e4m3 | group |

Activation scales: per-tensor (static), per-token (dynamic), per-group FP32 (group multiple of
32, e.g. 128 for DeepSeek-style blockscale) and MX E8M0 g32.

NotImplementedError (tested in `test_layer_f8_unsupported_raises`):

- NVFP4 (FP4 weights with group-16 e4m3 scales);
- FP / MX weights with f16 / bf16 activations (priority 4 was not cheap: the FP16 kernel's
  dequant path has no FP-weight decoder, and FP8 activations cover the use case);
- 8-bit integer weights with zero points under INT8 activations;
- per-group activation scales when K is padded to the tile;
- FP4 activations with FP32 group scales;
- MoE (`gemm_type != DENSE`);
- INT8 activations with K > 131072 (INT32 partial sums stop being exact).

## Decisions and layouts

- **One new kernel, mode-templated.** `MfmaF8Gemm<Traits, Tile>` shares the structure of the
  FP16 kernel (register-prefetched B, LDS-staged A, split-K with a counter) but every compile-time
  choice comes from the plan: A format, B mode (`B_NATIVE` / `B_LUT` / `B_INT_ARITH`), MFMA K (128,
  64 for INT8, 32 when scale groups are finer than 128), weight / activation scale modes
  (`NONE`, `FLUSH` = FP32 scale applied to a per-group accumulator, `HW` = E8M0 scale operand,
  `E8M0_FLUSH`).
- **MFMA fragment layouts** (probed in `test_mfma_f8_layout.py`, lg = lane / 16):
  8-bit operands of the scaled 16x16x128: k = (j / 16) * 64 + lg * 16 + j % 16 ("L128");
  FP4 / FP6 operands: k = 32 lg + j ("TRUE"); i8 16x16x64: k = 16 lg + s; x32 MFMAs:
  k = 8 lg + s ("L32"); C: column l % 16, row (l / 16) * 4 + r. E8M0 scale operand: byte 0,
  127 = 1.0. Weights are repacked offline to these lane orders so each lane loads B with
  16-byte buffer loads.
- **Accumulation precision.** With one 8-bit operand the MFMA accumulates with ~1e-4..2e-4
  relative error (vs ~1e-5 for FP16 MFMAs); tests use 5e-3 (bf16 out) / 1e-3 (fp16 out) against
  a reference built from the layer's own dequantized weights and quantized activations.
- **W4A8 via a byte LUT.** INT1-5 / zero-point values are mapped to e4m3 bytes with one
  `v_perm_b32` per 4 values (all INT4 values are exact in e4m3); group scales are applied per
  flush group on an FP32 accumulator, so the MFMA always runs at the 16x16x128 f8 rate. The
  `v_perm` selector bytes needed an explicit mask (unmasked high selector bits pick the
  constant-fill results instead of table bytes).
- **INT8 x INT4** uses integer arithmetic (sign-extend nibbles to bytes), not a LUT.
- **Tiles per M bucket** (`F8_M_INTERVALS`, 9 buckets): decode (M <= 8) uses no LDS, one
  16-row tile and 4 stages; mid M uses narrow blocks (32 or 64 rows x 64 columns, one 16-column MFMA
  tile per wave, 3 stages: latency-bound, so more blocks win); large M uses 128 x 128 blocks
  (8x2 MFMA tiles per wave) for FP8 / INT8 / LUT B, 64 x 128 for native B with FP32 block
  scales, 2 x 2 waves of 64 x 64 for FP4 activations, and 2 stages where registers are tight
  (LUT and INT-arithmetic B). Every bucket compiles without VGPR spills (checked with
  `llvm-readelf --notes` on the JIT cache).

## Measured MFMA peaks (single kernel, all CUs)

| MFMA | TFLOPS |
|---|---|
| 16x16x128 f8 | 4206 |
| 16x16x128 f4 | 8023 |
| 16x16x128 f8 x f4 | 4395 |
| 32x32x64 f8 | 4516 |
| 32x32x64 f4 | 8394 |

## Iterations

GEMM-only, bf16 output, us.

1. **Functional v1** (77d7e24). All formats correct, int32 partial accumulators for every
   mode, tile_n outer loop. W4A8 int4 g128 4096^3: 236 us.
2. **Direct-flush accumulators, tile_m outer loop, M buckets, L2 swizzle** (61ab6a8). MFMAs
   accumulate straight into the FP32 C fragment (or a per-flush temporary when a scale group
   ends inside the chunk); INT32 partials only for unscaled INT8. Looping tile_m outside
   tile_n keeps one A fragment live and removed the W4A8 spills. Group-M / XCD block remap.
   W4A8 4096^3: 236 -> 102 us; FP8 x MXFP4 4096^3: 82 us; FP8 4096^3: 95 us. The L2
   swizzle alone gave ~8 % at M=4096.
3. **Spill fixes in the large-M heuristic.** Two large-M tiles spilled (found with
   `llvm-readelf`): FP8 block-128 in 8x2 / 3 stages (362 VGPRs) and INT8 x INT4 with 3 stages
   (85 VGPRs). FP8 block-128 now uses 64 x 128 blocks (4x2 tiles per wave), INT-arithmetic B 2 stages.
   FP8 block-128 4096^3: 1240 -> 141 us; INT8 x INT4 4096^3: 310 -> 108 us.
4. **LDS-staged activation group scales.** Per-group FP32 activation scales were loaded per
   lane, per 16-row tile and per flush (4 buffer loads each). They are now loaded once per
   block row and chunk with the A chunk, written next to A in LDS and read as one 16-byte LDS
   load per tile. FP8 block-128 x g128 activations: 4096^2 M=1024 40.2 -> 35.4 us, M=4096
   141 -> 125 us; 8192 x 28672 (N x K) M=256 145 -> 127 us, M=4096 1763 -> 1673 us.
5. **Deferred flush scaling** (warp tiles of <= 8 MFMA tiles). The FP32 scale of tile_m - 1 is
   applied after the MFMAs of tile_m are issued, so the VALU work no longer waits on the MFMA
   result. W4A8 8192 x 28672 M=256: 168 -> 127 us; 4096^2 M=1024: 39 -> 35 us. Applying it
   to the 16-tile (8x2) large-M tiles spilled 5 VGPRs and slowed W4A8 8192 x 28672 M=4096
   1124 -> 1242 us, so those tiles keep the one-accumulator order (back to 1123 us,
   0 spills).

Current GEMM-only state (us):

| N x K | M | FP8 ch | FP8 blk128 | W4A8 int4 | INT8 x INT4 | FP8 x MXFP4 | MXFP4 x MXFP4 | INT8 |
|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 | 16 | 7.1 | 7.8 | 8.4 | 8.0 | 7.2 | 10.0 | 7.4 |
| 4096 x 4096 | 256 | 12.3 | 17.3 | 18.1 | 17.2 | 12.5 | 19.4 | 14.0 |
| 4096 x 4096 | 1024 | 30.6 | 35.1 | 35.3 | 35.8 | 29.7 | 43.8 | 32.1 |
| 4096 x 4096 | 4096 | 96.4 | 123.1 | 104.0 | 108.2 | 81.1 | 103.9 | 85.6 |
| 8192 x 28672 | 16 | 36.0 | 37.5 | 39.2 | 34.7 | 29.5 | 26.2 | 36.1 |
| 8192 x 28672 | 256 | 96.6 | 127.4 | 125.5 | 137.0 | 98.0 | 127.0 | 108.0 |
| 8192 x 28672 | 1024 | 248.9 | 431.6 | 285.4 | 310.2 | 209.7 | 312.4 | 257.9 |
| 8192 x 28672 | 4096 | 1082 | 1677 | 1123 | 1223 | 902 | 1228 | 1244 |

## Dead ends

- `__launch_bounds__(256, 1)` (more registers): no gain; the 2-blocks/CU occupancy matters more.
- Larger per-wave tiles (2x2 waves of 8x4, 8x8) at large M: spills or lower occupancy, slower.
- `lds_a=False` at large M: every wave re-reads A from L2; slower in every large-M bucket.
- Preloading the next A fragment from LDS before the MFMAs: no gain, more VGPRs.
- 128-row tiles for FP8 block-128 even with LDS activation scales: 8x2 / 3 stages still spills
  261 VGPRs (4096^3 619 us vs 123 us).
- There is no unscaled 16x16x128 f8 builtin on gfx950; the scaled one with neutral (127)
  scales runs at the same rate, so it is used for all 8-bit operands.
- Deferred flush on 16-tile warp tiles (see iteration 5).
- A tile-sweep script bug (bucket upper bound used as M) made one early sweep pick tiles for
  the wrong bucket; re-swept.

## Benchmark vs ROCm implementations (bench_baselines, 2026-10-28)

`benchmarks/rocm/results/humming_f8_2026-10-28.{csv,md}`; bf16 output, weights rotated over
enough copies to defeat the MALL. Cells: Humming us / best same-format ROCm us (which) =
speedup (> 1 means Humming is faster). Humming entries are GEMM-only (activation quantization
excluded, as for the AITER entries, which take pre-quantized inputs).

| format | shape | M=1 | M=16 | M=1024 | M=4096 |
|---|---|---|---|---|---|
| FP8 per-token x FP8 per-channel | 8B o_proj | 8.2 / 8.3 a8w8 = 1.00x | 9.3 / 8.4 bpreshuffle = 0.90x | 32.7 / 25.3 bpreshuffle = 0.77x | 103 / 59.0 scaled_mm = 0.57x |
| (vs scaled_mm, AITER a8w8 triton / CK / bpreshuffle) | 70B gate_up | 78.9 / 82.1 scaled_mm = 1.04x | 86.2 / 83.9 a8w8 = 0.97x | 607 / 360 scaled_mm = 0.59x | 2358 / 1437 CK = 0.61x |
| | 70B down | 41.9 / 41.7 scaled_mm = 1.00x | 44.9 / 42.3 scaled_mm = 0.94x | 272 / 181 scaled_mm = 0.66x | 1124 / 638 bpreshuffle = 0.57x |
| FP8 g128 x FP8 block-128 | 8B o_proj | 9.3 / 11.7 = 1.27x | 9.8 / 11.2 = 1.14x | 40.0 / 29.6 = 0.74x | 146 / 86.9 = 0.60x |
| (vs AITER blockscale triton / CK / bpreshuffle) | 70B gate_up | 81.8 / 92.5 = 1.13x | 91.7 / 94.5 = 1.03x | 903 / 848 = 0.94x | 3508 / 3310 = 0.94x |
| | 70B down | 45.3 / 106 CK = 2.34x | 49.7 / 108 CK = 2.16x | 453 / 440 = 0.97x | 1673 / 1678 = 1.00x |
| FP8 x MXFP4 (W4A8 fp4) | 8B o_proj | 8.0 / 22.4 = 2.81x | 8.9 / 12.3 = 1.37x | 30.8 / 93.3 = 3.03x | 88.0 / 133 = 1.51x |
| (vs AITER a8wfp4) | 70B gate_up | 42.1 / 102 = 2.43x | 48.8 / 134 = 2.74x | 521 / 954 = 1.83x | 2005 / 3252 = 1.62x |
| | 70B down | 34.6 / 153 = 4.41x | 41.6 / 67.0 = 1.61x | 232 / 646 = 2.78x | 911 / 1794 = 1.97x |
| FP8 x INT4 g128 (W4A8 int) | 8B o_proj | 8.1 / 8.5 W4A16 = 1.06x | 9.6 / 11.7 W4A16 = 1.22x | 35.4 / 54.8 W4A16 = 1.54x | 107 / 133 a8wfp4 = 1.24x |
| (vs best 4-bit-weight kernel: AITER a8wfp4 / a16wfp4, Humming W4A16) | 70B gate_up | 46.2 / 53.7 a16wfp4 = 1.16x | 53.9 / 95.5 W4A16 = 1.77x | 625 / 954 a8wfp4 = 1.53x | 2405 / 3252 a8wfp4 = 1.35x |
| | 70B down | 38.3 / 43.4 W4A16 = 1.13x | 46.3 / 53.5 W4A16 = 1.16x | 299 / 543 W4A16 = 1.82x | 1152 / 1794 a8wfp4 = 1.56x |
| MXFP4 x MXFP4 | 8B o_proj | 7.4 / 7.2 afp4wfp4 = 0.97x | 11.1 / 7.0 afp4wfp4 = 0.63x | 43.5 / 14.7 asm = 0.34x | 116 / 35.0 asm = 0.30x |
| (vs AITER afp4wfp4, a4w4 asm) | 70B gate_up | 42.5 / 45.7 asm = 1.07x | 53.1 / 45.9 asm = 0.86x | 655 / 211 asm = 0.32x | 2477 / 742 asm = 0.30x |
| | 70B down | 26.8 / 34.3 afp4wfp4 = 1.28x | 30.4 / 33.1 afp4wfp4 = 1.09x | 351 / 109 asm = 0.31x | 1285 / 338 asm = 0.26x |
| INT8 x INT8 | 8B o_proj | 8.5 / 8.5 = 0.99x | 9.4 / 9.9 = 1.06x | 34.4 / 35.5 = 1.03x | 93.9 / 100 = 1.07x |
| (vs AITER a8w8 int8) | 70B gate_up | 79.1 / 102 = 1.29x | 86.5 / 87.6 = 1.01x | 556 / 643 = 1.16x | 2134 / 2472 = 1.16x |
| | 70B down | 41.7 / 59.1 = 1.42x | 43.4 / 67.4 = 1.56x | 278 / 324 = 1.16x | 1271 / 1213 = 0.95x |
| INT8 x INT4 g128 | 8B o_proj | 8.0 / 8.5 = 1.06x | 9.2 / 9.9 = 1.08x | 36.5 / 35.5 = 0.97x | 113 / 100 = 0.88x |
| (vs AITER int8 W8A8, Humming W4A16) | 70B gate_up | 42.6 / 84.8 W4A16 = 1.99x | 48.0 / 87.6 int8 = 1.83x | 672 / 643 int8 = 0.96x | 2599 / 2472 int8 = 0.95x |
| | 70B down | 37.2 / 43.4 W4A16 = 1.17x | 44.2 / 53.5 W4A16 = 1.21x | 325 / 324 int8 = 1.00x | 1247 / 1213 int8 = 0.97x |

The dynamic per-token FP8 path end to end (`humming_w8a8_fp8_e2e`, quantization kernel
included) adds 6-8 us at small M (e.g. 70B down M=1: 41.9 -> 49.1 us).

Summary: decode and small M (M <= 16) are at parity or faster for every format (except
MXFP4 x MXFP4 at M=16 on small N). W4A8 (int4 and MXFP4 weights) is 1.1-4.4x faster than
any 4-bit-weight ROCm kernel at all M, and INT8 W8A8 beats AITER int8 at nearly every point.
Large-M FP8 per-channel (0.57-0.66x of hipBLASLt / AITER CK) and MXFP4 x MXFP4 (0.26-0.34x
of the a4w4 asm kernel) remain the main gaps.

## Gaps and future work

- FP8 per-channel large M plateaus at ~1.7-1.8 PFLOPS (8192 x 28672) vs ~2.8 PFLOPS for hipBLASLt
  (`torch._scaled_mm`) and the 4206-4516 TFLOPS MFMA peak; the kernel uses 16x16 MFMAs, at most
  128 x 128 block tiles and register-prefetched (not LDS-staged) B. Closing this needs larger
  LDS-staged tiles, i.e. a large-M restructuring like the one being done for the FP16 kernel.
- FP8 block-128 is ~1.5x slower than per-channel at large K (two FP32 scales per 128 K and a
  64-row tile); it matches AITER's best blockscale kernel (`gemm_a8w8_blockscale_bpreshuffle`) on the 70B shapes
  at large M (0.94-1.00x) and is 1.0-2.3x faster at M <= 16, but trails it on 8B o_proj at
  large M (0.60-0.74x).
- MXFP4 x MXFP4 at large M trails AITER's a4w4 asm kernel (~3x at M >= 1024; the asm kernel
  reaches ~5.7 PFLOPS on 70B down at M=4096, ~70 % of the f4 MFMA peak, vs ~1.5 PFLOPS here).
  The FP4 x FP4 path would need 32x32x64 MFMAs and larger LDS-staged B tiles.

## AITER baseline notes

- `gemm_a8w8_blockscale` (Triton) crashes in `TritonAMDGPUCanonicalizePointers` for M=1 (and
  with the M <= 8 config); the harness pads M=1 to 2 rows and uses the M=16 config.
- `gemm_a8w8_bpreshuffle` dispatches 70B gate_up (all M) and 70B down (M=1) to FlyDSL kernels
  via its tuned config and fails with "Unsupported `flydsl` version: expected >=0.1.8, got
  0.1.3.1"; those cells are errors in the results.
- CK entries (`gemm_a8w8_CK`, `gemm_a8w8_blockscale`) JIT-build into
  `benchmarks/rocm/.aiter_jit` (set `AITER_JIT_DIR`, otherwise a GLIBCXX mismatch).

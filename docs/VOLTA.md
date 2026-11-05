# Humming on Volta (Tesla V100, sm_70)

Status: working, tested, measured on one Tesla V100-PCIE-32GB (CUDA 12.9, PyTorch 2.10 + cu128, driver 575).
Volta is one generation older than Humming's supported floor (Turing, sm_75). This branch makes `float16`
activations with 1-8-bit weights run on it; `int8`, `fp8` and `fp4` activations need tensor-core formats Volta
does not have and stay unsupported (the kernels skip them exactly as they skip other formats on other cards).

## What Volta lacks, and what replaces it

| Turing instruction the kernels use | On Volta | Replacement |
| --- | --- | --- |
| `ldmatrix` (tile load into the mma fragment layout) | missing | `volta_ldmatrix` / `_trans`: fetch the row address from the lane that holds it, plain `ld.shared` (`utils/ptx/volta.cuh`) |
| `mma.sync.m16n8k8` fp16 | missing; Volta has `mma.sync.m8n8k4` | see below |
| `__reduce_add_sync`, int8 / bf16 tensor cores, `cp.async` | missing | guard widened; int8 / bf16 activations not offered; the legacy load path (as on Turing) |

Volta's `m8n8k4` fragment layout was measured directly on the card rather than taken from the PTX manual: a 1.0
in one register half of A and one of B at a time, all 16,384 pairs, recording which outputs light up. One instruction computes four independent 8x8x4 products, one per
group of 8 lanes (lanes 4i..4i+3 and 16+4i..16+4i+3). A Turing 16x8x8 is exactly four of them, so one Volta
instruction replaces one Turing instruction:

- **Decode path** (8-row tiles, 1-8 tokens, `float16` x int4/int8 weights, no zero point; `Ctx::kVoltaDecode`):
  the loaders put each lane's operands straight into the group layout. `loader_b` loads the two Turing lanes' packed
  words that hold this lane's weight row, `volta_dequant_half` converts only the needed half, `loader_a` reads the
  lane's four activation k values with one 8-byte shared load, the column scale comes from lane 4j with one shuffle
  per k iteration, and the mainloop issues one `mma.sync.m8n8k4` per k step with no shuffles. The two K halves live
  in partner groups and are added once after the K loop (`volta_finalize_tile`, 24 shuffles per tile), which also
  restores Turing's accumulator layout for the unchanged epilogue.
- **Prompt path** (16-row tiles, same conditions; `Ctx::kVoltaPrefill`): the same mapping, both source lanes' words
  kept (tile 2i = column j, tile 2i+1 = column 8+j), two weight registers per k step.
- **Everything else on Volta** (other weight widths, fp weights, zero points): `volta_mma_m16n8k8_fast`
  takes Turing-layout fragments, moves them into the group layout with six shuffles, and issues one `m8n8k4`.
  Correct and tested.
- `volta_mma_m16n8k8_f32` is a shuffle + fp32-FMA reference for both, kept for checking.

The Python side: `tune/sm70.py` (Turing's heuristics with fp16 only, 32x64 / 16x64 warp tiles, CTAs per SM from the
register budget, 64-column warp tiles kept for the prompt path, shorter K slices for small-N tables), the fp16 tensor-core rate and `float16` admitted from sm_70
in `csrc/device_info.cpp`, `kernel/humming.py`, `testing/device.py`, and the register counts for the two Volta paths in `config/mma.py`
(mirrored from the C++ switches `Ctx::kVoltaDecode` / `kVoltaPrefill`).

## Verification

On one Tesla V100-PCIE-32GB, CUDA 12.9, PyTorch 2.10 + cu128:

- `tests/kernels/humming`: 1524 passed, 0 failed, 532 skipped (formats Volta lacks; `test_f16_accumulation` skips on
  sm_70 because the Volta path accumulates in fp32 only). With `HUMMING_TEST_TUNING_SOURCE=sampled20`,
  `test_datatype.py` + `test_shape.py`: 32 passed, 0 failed.
- An independent forced-configuration sweep: 4,435 valid tile configurations from the repo's own generators, int4 and
  int8 weights, four shapes, M = 1 to 256 (31,045 runs), each against the dequantized weights: 0 failures, worst
  relative error 1.8e-3 (tolerance 1e-2). Forced configurations matter: the auto-chosen ones once took a different
  path and hid a register-count bug. Warp tiles whose Volta accumulators cannot fit in registers are rejected by the
  register-budget check; `--register-usage-level=10` is not used for sm_70, because with it ptxas turned one heavily
  spilling tile into an illegal shared-memory read.
- No change for other architectures: the test suite's kernels compiled for sm_75, sm_80, sm_86 and sm_89 from
  upstream `main` and from this branch have byte-identical cubins: all 916 / 2111 / 2111 / 3016 distinct kernels. In an earlier run two sm_89
  `test_zero_point` uint3 kernels differed; upstream itself compiles those nondeterministically. Every Volta branch is
  under `__CUDA_ARCH__ < 750` or `sm_version == 70`, and non-Volta builds evaluate no new expression.
- The `ldmatrix` and `m16n8k8` stand-ins were checked against a CPU model of both instructions: bit-for-bit for
  `ldmatrix`, 2e-5 for the multiply.

## Speed

GPU time inside a CUDA graph (as vLLM runs it; Python launch overhead is CPU-bound below ~15 us), SM clock locked at
1380 MHz, microseconds per call, median of 7 replays, random normal activations. Every method rotates over enough copies
of its weights (96 MiB or more; the V100's L2 is 6 MB) that each call reads them from memory, as in a real model --
the same rule as the llama.cpp numbers, which are its own `Q4_0` / `Q8_0` kernels on the same card and clock from the
kernel bake-off (github.com/christopherrobertbrooks-tech/volta-kernel-bakeoff; they include llama.cpp's conversion of
the activations to 8 bits, about 2 us per call). Humming `int4 g32` keeps one fp16 scale per 32 weights, like `Q4_0`;
`int4` and `int8` keep one per output channel. A second, independent timing harness reproduced the Humming int4 and int8 numbers within 2% and llama.cpp's
640 x 2560 one-token figure (6.81 us); validated first against the bake-off's records, it matched llama.cpp at a
0.3% median difference and cuBLAS at 0.8%, with 11 of 90 cuBLAS cells reading 10-14% high for reasons it could not
pin down. The `int4 g32` column and the other llama.cpp cells come from one harness only.

| table (n x k) | tokens | cuBLAS fp16 | Humming int4 | Humming int4 g32 | llama.cpp Q4_0 | Humming int8 | llama.cpp Q8_0 |
| --- | --: | --: | --: | --: | --: | --: | --: |
| 12288 x 2560 | 1 | 99.3 | 30.9 | 35.0 | 32.4 | 52.5 | 58.2 |
| 12288 x 2560 | 8 | 106.1 | 32.1 | 36.0 | 53.1 | 54.1 | 67.4 |
| 8192 x 2048 | 1 | 47.2 | 21.5 | 23.8 | 20.4 | 31.6 | 32.1 |
| 8192 x 2048 | 8 | 66.9 | 21.8 | 24.2 | 32.3 | 32.2 | 39.7 |
| 6144 x 2048 | 1 | 42.5 | 19.9 | 21.9 | 16.4 | 29.5 | 25.3 |
| 6144 x 2048 | 8 | 44.5 | 20.4 | 22.2 | 25.9 | 29.9 | 31.4 |
| 640 x 2560 | 1 | 9.8 | 13.6 | 14.5 | 6.8 | 14.5 | 7.3 |
| 640 x 2560 | 8 | 14.4 | 13.8 | 14.7 | 10.3 | 14.6 | 10.0 |
| 512 x 2048 | 1 | 8.5 | 12.9 | 13.8 | 7.2 | 14.1 | 6.7 |
| 512 x 2048 | 8 | 10.6 | 13.0 | 14.0 | 10.4 | 14.1 | 9.5 |
| 2048 x 512 | 1 | 6.3 | 5.8 | 6.2 | 6.6 | 7.4 | 6.0 |
| 2048 x 512 | 8 | 7.6 | 5.8 | 6.2 | 9.7 | 7.5 | 8.8 |

- **Big tables (N = 6144-12288):** 2-3x faster than cuBLAS fp16 at 1-8 tokens. Against llama.cpp at the same scale
  grouping (`int4 g32` vs `Q4_0`): 8-34% slower at one token, 1.2-1.5x faster at eight; per-channel `int4` is within
  5% of `Q4_0` at one token on the largest table.
- **Expert-sized tables (N = 512-640):** about 2x slower than llama.cpp and 1.4-1.6x slower than cuBLAS at one
  token, and about 1.3x slower than llama.cpp at eight. The cause is not profiled yet. `2048 x 512` is the exception
  (faster than both).
- Timing every call on the same weights instead leaves small tables in L2, where they look 10-30% faster than they
  are; an earlier version of this table made that mistake.

**Prompt sizes** go through the 16-row tiles, which have the same native path (`Ctx::kVoltaPrefill`: one `m8n8k4`
per m tile, n tile and k step, no shuffles; 16x64 and 32x64 warp tiles). It is correct, but Volta cannot keep up with
cuBLAS fp16 as the token count grows (GPU time, same clock and rule, per-channel int4):

| table | 64 tokens: Humming / cuBLAS | 256 tokens: Humming / cuBLAS |
| --- | --: | --: |
| 12288 x 2560 | 185 / 169 | 634 / 323 |
| 8192 x 2048 | 108 / 95 | 359 / 120 |
| 6144 x 2048 | 82 / 70 | 268 / 119 |

Measured from 8 to 128 tokens with the same rule (`int4`, microseconds), Humming leads cuBLAS fp16 on big tables up
to 16 tokens; cuBLAS overtakes it from 24 tokens on 8192 x 2048 and 6144 x 2048 and from 48 tokens on 12288 x 2560,
by 8-18% up to 64 tokens:

| table | 8 | 16 | 24 | 32 | 48 | 64 | 128 | tokens |
| --- | --: | --: | --: | --: | --: | --: | --: | --- |
| 12288 x 2560 | 32 / 106 | 61 / 111 | 97 / 106 | 99 / 108 | 185 / 167 | 185 / 170 | 328 / 264 | Humming / cuBLAS |
| 8192 x 2048 | 22 / 69 | 29 / 66 | 64 / 57 | 66 / 58 | 106 / 94 | 108 / 95 | 183 / 109 | Humming / cuBLAS |
| 6144 x 2048 | 21 / 44 | 28 / 39 | 53 / 45 | 54 / 45 | 80 / 69 | 82 / 70 | 149 / 133 | Humming / cuBLAS |

cuBLAS reads fp16 weights here, four times the memory of int4. Two structural reasons for the crossover: Volta's
`m8n8k4` does half the work of Turing's `m16n8k8` per instruction, so instruction overhead per FLOP doubles; and the
warp tiles that fit Volta's registers reuse each dequantized weight for one or two row tiles where Turing's 64x64 tile
reuses it for four. A Volta dispatch rule for prompt-sized M belongs in the heuristics; this branch reports the numbers
and leaves the choice to the maintainers.

Where the time goes (Nsight Compute, 8192x2048 int4, 1 token): 16 warps per SM at most (113-128 registers per
thread), about 3 warps per scheduler, an instruction issued every 9-11 cycles; the top stall reasons are
`no_instruction` (the 48 KB kernel body and the instruction cache), `long_scoreboard` (global loads) and `barrier`.
The multiply itself was never the limit: the shuffle and native paths take the same time on the big tables.

## Lessons that cost time

- A runtime index into a register array (`words[nh * 2]`, `qb[2 * i + nh]`) compiles to local memory: 704 bytes of
  stack and a 2x slowdown, with no warning. Use selects.
- `nvrtc` 12.9 still compiles sm_70 (with a deprecation warning); `ptxas` names the exact instructions a target lacks.
- Timing a 20 us kernel from Python measures Python. Capture a CUDA graph.
- Time small tables with cycled weights: the V100's L2 is 6 MB.

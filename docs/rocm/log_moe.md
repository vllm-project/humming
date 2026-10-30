# Humming MoE GEMM on ROCm (gfx950 / MI355X)

Branch `rocm-moe`. Adds the MoE (grouped) weight-only GEMM to the ROCm MFMA backend, usable through
the same `humming.ops.humming_gemm` / `HummingLayer` interface that vLLM's `fused_humming_moe.py`
already drives on CUDA.

## Supported matrix

| | supported on ROCm |
|---|---|
| activations | fp16, bf16 (16-bit only; FP8/MX activations are a separate work item) |
| weights | INT1 ... INT8, symmetric or with zero point |
| scales | per channel or per group (group 32 / 64 / 128 ... ) |
| layouts | `indexed` (vLLM `sorted_token_ids` / `expert_ids` contract), `grouped_contiguous`, `grouped_masked` |
| GEMMs | w13 (gate+up, top_k > 1 gather) and w2 (top_k = 1, weighted sum done outside) |
| extras | per-expert bias, unpadded N, split-K, CUDA(HIP)-graph capture, `moe_align_block_size` (HIP kernel) |

## Design

* `humming/include/humming/kernel/humming_mfma_moe.cuh`: derives from the dense `MfmaWeightOnlyGemm`
  and shadows only `load_chunk` / `mainloop`. **No edit to `humming_mfma.cuh`.**
  * Every M block reads its expert id (`expert_ids[block]`, or the group of a contiguous / masked
    layout) and offsets the B / scale / zero-point / bias base pointers by that expert.
  * A is gathered through a per-row byte offset (`a_row_bytes[row]`): for `indexed` the row is
    `sorted_token_ids[i] / top_k` (w13) or `sorted_token_ids[i]` (w2). Padding rows (id >= numel)
    point at row 0 and are masked at the store. C is scattered by the same row map.
  * Split-K reuses the dense workspace / reduction.
* Launcher: `csrc/launcher/mfma_moe.h`, dispatched through the `MFMA_MOE_TYPE` symbol so the dense
  launcher path is unchanged.
* Heuristics (`humming/tune/gfx950.py`): `select_moe_tile(valid_shape_m, num_experts)` picks the
  tile from rows per expert `r = ceil(valid_shape_m / E)`, and `moe_m_intervals` exposes the
  intervals so vLLM's `block_m = tuning_config[interval].block_shape[0]` contract holds (block_m is
  taken from w13 and reused for w2).
* `moe_align_block_size` (`humming/ops/moe.py`): on ROCm a one-launch HIP kernel
  (`humming/include/humming/kernel/moe_align.cuh`, `humming/kernel/moe_align.py`); the torch
  implementation stays as `moe_align_block_size_torch` (fallback for CPU, non-int ids, E > 1024).
  * One workgroup of 1024 threads: LDS counts -> Hillis-Steele scan of per-expert block counts ->
    `expert_ids` by binary search -> LDS-atomic scatter. Row order within an expert is arbitrary
    (as in vLLM's op), handles `expert_map` and `ignore_invalid_experts`.

## Iterations (numbers are one MoE block: align + w13 + SiLU*mul + w2 + top-k sum, us)

1. **First working kernel + torch align** (commit 2173394): correct on all layouts, but the torch
   align cost 73-178 us (~25 small kernels) which dominated decode (qwen3 T=1 GEMMs take ~13 us).
2. **HIP align** (699c5ba): 3.5-4.5 us at decode, ~10 us at 1k tokens x top8, ~55 us at 8192 x 8.
   With that and Triton glue kernels (SiLU-and-mul, weighted top-k sum, ~3 us each instead of
   ~10 us torch chains), Qwen3-30B-A3B bf16 W4:

   | T | humming | aiter a16wi4 | aiter bf16 | triton wna16 |
   |---|---|---|---|---|
   | 1 | 21.6 | 25.0 | 28.0 | 43 |
   | 16 | 76.9 | 84.5 | | |
   | 64 | 109 | 93.4 | | |
   | 1024 | 194 | 300.9 | 296 | 417 |

3. **Tile retune** (5d9a8af). Sweep of `tiles_m / tiles_n / lds_a / scale_b` per rows-per-expert:
   * `lds_a` (stage A through LDS) hurts when a block holds ~1 valid row (worst on Mixtral w2,
     K = 14336) and helps once an expert has several rows -> direct A loads for
     `valid_shape_m <= 16`.
   * `scale_b=False` is better at tiles_m 4, tiles_m 8 needs `scale_b`. W8 has the same preferences.
   * Buckets: vm <= 16 -> (1,1); r <= 8 -> (1,1,lds_a); r <= 16 -> (2,1,lds_a);
     r <= 256 -> (4,2,lds_a); else (8,2,lds_a,scale_b).

   | shape (W4, w13+w2) | before | after |
   |---|---|---|
   | qwen3 T=64 | 101 | 71 |
   | mixtral T=64 | 411 | 186 |
   | dsv3 T=16 | 608 | 482 |
   | qwen3 T=2048 | 299 | 285 |

4. **DeepSeek-V3 correctness bug** found by the full sweep (rel_err 0.99): not the GEMM, but
   `quant_weight.cuh` (weight quantization used by `load_from_unquantized`) computed its global
   offset in `uint32_t`; the DeepSeek-V3 w13 tensor has 256 x 4096 x 7168 = 7.5 G elements, so
   experts >= ~146 were quantized from the wrong data. Fixed with 64-bit offsets in
   `quant_weight.cuh` and the same pattern in `pack_weight.cuh` (pack / unpack). After the fix every
   expert dequantizes to the reference within the expected int4 error (0.10 rel), the kernel matches
   its own dequant at 1.6e-3.

## Results (`benchmarks/rocm/results/humming_moe_2026-10-28.{csv,md}`)

`benchmarks/rocm/bench_moe.py`, GPU 5, HIP-graph replay, best of 5, weight copies rotated so >= 1 GiB
is touched per cycle. Baselines: vLLM Triton `fused_moe` wna16 (vLLM default config and a tuned
config sweep), AITER `a16wi4` int4 MoE (bf16 only, group 32), Triton and AITER 16-bit MoE. All use
the same HIP align and glue so only the GEMMs differ.

bf16, W4 group 128; "best alt" = fastest non-Humming quantized MoE (AITER a16wi4 at every point here):

| model | T | Humming W4 | best alt W4 | speedup | best 16-bit | vs 16-bit |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 1 | 21.1 | 24.8 | 1.18x | 25.9 | 1.23x |
| Qwen3-30B-A3B | 16 | 63.3 | 85.4 | 1.35x | 160.6 | 2.54x |
| Qwen3-30B-A3B | 64 | 79.1 | 95.4 | 1.21x | 209.0 | 2.64x |
| Qwen3-30B-A3B | 2048 | 316.6 | 396.4 | 1.25x | 401.4 | 1.27x |
| Qwen3-30B-A3B | 8192 | 1069.0 | 1325.4 | 1.24x | 1086.6 | 1.02x |
| Mixtral-8x7B | 1 | 70.2 | 161.1 | 2.29x | 204.8 | 2.92x |
| Mixtral-8x7B | 16 | 172.5 | 252.1 | 1.46x | 482.5 | 2.80x |
| Mixtral-8x7B | 64 | 209.4 | 298.3 | 1.42x | 496.8 | 2.37x |
| Mixtral-8x7B | 2048 | 1914.2 | 2751.4 | 1.44x | 1775.8 | 0.93x |
| Mixtral-8x7B | 8192 | 6923.2 | 10650.2 | 1.54x | 5781.0 | 0.84x |
| DeepSeek-V3 | 1 | 51.7 | 69.3 | 1.34x | 136.5 | 2.64x |
| DeepSeek-V3 | 16 | 476.6 | 595.6 | 1.25x | 1469.7 | 3.08x |
| DeepSeek-V3 | 64 | 1036.9 | 1190.0 | 1.15x | 3215.6 | 3.10x |
| DeepSeek-V3 | 2048 | 2817.4 | 3504.5 | 1.24x | 4088.8 | 1.45x |
| DeepSeek-V3 | 8192 | 8552.5 | 11461.8 | 1.34x | 7573.9 | 0.89x |

fp16 has no AITER int4 path; there Humming W4 is 1.8-4.1x the best Triton wna16. W8 vs the tuned
Triton int8 wna16: 1.2-2.4x. At large prefill on the dense-ish Mixtral / DeepSeek shapes the 16-bit
AITER MoE wins by 3-16% from T=1024 (bf16): the MFMA mainloop is not yet at the dense large-M efficiency (the dense
large-M work happens on a separate branch and would carry over through the shared base class).

## Dead ends

* Torch `moe_align_block_size`: 73-178 us, replaced by the HIP kernel.
* Unrolling the align kernel loads (8 independent loads per thread): no gain, the kernel is LDS
  atomic / scattered store bound past ~1k rows.
* Skipping the global loads of padding A rows (buffer OOB trick): no measurable gain, reverted.
* vLLM's FlyDSL MoE backend as a baseline: needs a newer aiter (`moe_2stage_a16wmix` missing in the
  installed one); flydsl 0.2.3 had to be shadowed through `benchmarks/rocm/.pydeps` (gitignored).
* torch glue (SiLU-and-mul, top-k weighted sum as torch op chains) cost ~10 us per MoE at decode,
  replaced by two small Triton kernels in the benchmark (vLLM uses its own fused ops).

## What vLLM needs (not modified here)

1. `fused_humming_moe.py::_supports_current_device` requires `platform.is_cuda()` and capability
   (7,5): allow `platform.is_rocm()` with gfx950 (`humming.tune` raises on other ROCm archs).
   The dense Humming linear kernels (`kernels/linear/mxfp4|mxfp8/humming.py`) have the same gate,
   but those are FP4/FP8 weights and out of scope here.
2. Alignment: vLLM's `_C` `moe_align_block_size` works on ROCm; `humming.ops.moe_align_block_size`
   is an equivalent alternative with the same buffers.
3. block_m contract: unchanged; ROCm tuning configs carry `block_shape = [block_m, block_n, 128]`
   for every interval of `moe_m_intervals`, so `_fixup_moe_tuning_config` (caps block_k at 128,
   only touches `warp_shape` when present) is a no-op.
4. `compute_config`: `use_f16_accum` / `use_batch_invariant` are accepted when False (vLLM default);
   `VLLM_HUMMING_USE_F16_ACCUM=1` or `VLLM_BATCH_INVARIANT=1` raise `NotImplementedError` on ROCm.
5. Activation (`apply_activation`) and `moe_fused_mul_sum` (Triton) run on ROCm unchanged; the
   `locks` argument is accepted.

## Limitations

* 16-bit activations only.
* HIP align is a single workgroup: ~55 us at 64k routed rows (8192 tokens x top8), ~5% of the
  Qwen3 layer there; E <= 1024 (falls back to torch above).
* No auto-tuning beyond the heuristics; large prefill on Mixtral / DeepSeek-V3 is 3-16% behind the
  16-bit AITER MoE (bf16).
* Weight quantization of > 2^32-element tensors was broken before the `quant_weight.cuh` /
  `pack_weight.cuh` fix (affects any platform using these kernels); no unit test covers it because it
  needs > 8 GiB tensors, verified with the DeepSeek-V3 w13 shape instead.

## After merging rocm-perf (2026-10-28, branch rocm)

* `MfmaMoeGemm` copies the dense `load_chunk` / `mainloop`, so it had to follow rocm-perf's direct-to-LDS
  A path: one `__shared__` buffer per stage, `global_load_lds` with the swizzle applied on the global side
  (slot `thread_id + i * kNumThreads` is the same slot that `a_row_bytes[i]` already described), and
  `s_waitcnt vmcnt` + `s_barrier` before each chunk instead of the register-staged LDS store.
* Same-GPU A/B (best of 2x7, us, one MoE block) showed a real decode regression with the old 2-stage LDS
  tiles: with direct-to-LDS a 2-stage pipeline waits `vmcnt(0)` before every chunk. 3 stages for the 16- and
  32-row LDS tiles (`select_moe_tile`) fixes it; the 4x2 tile stays at 2 stages (3 stages: 0.95-1.02x).

| model / dtype / bits / tokens | pre-merge | merged, 2 stages | merged, 3 stages |
|---|---|---|---|
| mixtral fp16 W4 16 | 150.5 | 166.2 | 139.0 |
| mixtral bf16 W4 16 | 174.5 | 171.8 | 153.7 |
| mixtral fp16 W4 64 | 185.1 | 188.9 | 172.5 |
| qwen3 fp16 W4 16 | 56.0 | 61.0 | 55.3 |
| qwen3 bf16 W4 4 | 34.2 | 32.3 | 30.2 |
| mixtral fp16 W8 16 | 251.8 | 262.3 | 248.8 |

* Prefill gains come for free from the dense kernel work: W4 bf16 at 8192 tokens is 1.09x (Qwen3),
  1.14x (DeepSeek-V3), 1.17x (Mixtral) faster than before the merge
  (`benchmarks/rocm/results/humming_moe_merged_2026-10-28.{csv,md}`, Humming only).

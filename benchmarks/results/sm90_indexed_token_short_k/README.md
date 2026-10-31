# SM90 short-K indexed per-token W4A8 benchmark

**These are isolated GEMM kernel measurements, not end-to-end serving results.**
No model throughput, TTFT, communication, or multi-request latency improvement
is claimed. All inputs and routing are synthetic; no checkpoint is required.

[`benchmark.csv`](benchmark.csv) contains all 116 main-versus-patch comparison
rows, including hot/cold L2, uniform/skewed routing, nearby shapes and fallback
probes. `speedup_x = main_median_us / candidate_median_us`; latency reduction is
`100 * (main_median_us - candidate_median_us) / main_median_us`.

## Compared implementations

- Baseline: unchanged Humming main at
  `452e0bb15f72cc3db68af583d04425bc997dbaaa`.
- Candidate: that same source with the short-K indexed per-token policy in this
  PR. `policy_sha256` records the tested `humming/tune/sm90_policies.py` contents.
- Both arms obtain configurations through `generate_heuristics_configs`, using
  the main or patched policy. The benchmark does not force the proposed tile
  for shapes on which the policy falls back. `dispatch_changed` records whether
  the selected configuration changes.
- Activation/weight quantization, inputs, expert routing and output dtype are
  identical in each paired comparison. This is scheduling-only, not a precision
  change. No changes from the related open PRs are included in either arm.

## Environment and method

- NVIDIA H100 80GB HBM3, SM90, 132 SMs; GPU 0 for this patch matrix.
- PyTorch `2.13.0+cu130`, CUDA runtime `13.0`.
- FP8 E4M3 dynamic per-token activations with FP32 scales; MXFP4 E2M1 weights
  with group-32 E8M0 scales; BF16 output and FP32 accumulation.
- Indexed, unpacked-K expert down-projection. The primary fixture has 384
  experts, N=5120, logical K=288, caller-padded K=320, kernel-padded K=384.
  The caller's padding is zero-filled. The full expert weight bank is allocated.
- Already-expanded rows have independent per-expert inputs. Routing selects six
  distinct experts per original row and flattens the result for a `top_k=1`
  indexed W2 launch. Uniform routing uses random scores; skewed routing adds
  2.0 to the scores for the first 24 experts before selecting the top six.
- CUDA graphs contain 24 timed kernel calls. Hot-L2 and cold-L2 are separate
  measurements. Cold-L2 zeroes a 128 MiB flush buffer before **each** call;
  the flush and lock reset are outside the timed interval.
- Each cell uses five rounds (120 event samples per arm), with randomized
  interleaving of main/patch and hot/cold variants. Graph replay warmup is
  excluded. GPU clocks were not locked; samples are not independent end-to-end
  request repetitions.
- Timings exclude allocation, weight preparation, quantization, routing and
  alignment, JIT compilation, and cache flushing.
- Both outputs are checked against the existing `KernelTestRunner` reference
  using its normal tolerances. All reported rows passed. This does not claim
  bitwise equality or assess model quality.

`concurrency_equivalent` is a shape label, **not measured request concurrency**.
For the primary fixture, M=48/192/384/768 corresponds to eight verification rows
per request times six routed experts times C=1/4/8/16. Other rows leave that
field empty. No server or distributed TP8 benchmark was run for these numbers.

## Primary fixture: cold-L2, uniform routing

| Expanded rows | C-equivalent | Main (us) | Patch (us) | Kernel speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 1 | 42.464 | 31.936 | 1.330x | 24.79% |
| 192 | 4 | 92.144 | 73.040 | 1.262x | 20.73% |
| 384 | 8 | 134.256 | 106.176 | 1.264x | 20.92% |
| 768 | 16 | 180.016 | 138.736 | 1.298x | 22.93% |

Across these four sizes with both uniform and skewed routing, cold-L2 kernel
speedups range from 1.26x to 1.40x (20.7% to 28.5% lower latency).

## Coverage and limits

The CSV also includes M=24 on the primary fixture and nearby N/K/expert-count
combinations: N4096/K256/E384, N8192/K512/E384, N1024/K128/E384,
N4096/K384/E64, and N5120/K256/E128. Small/large-M probes include M6, M12,
M1536 and M3072. The `fallbacks` fixture name describes a probe set, not a
promise that every row falls back: heuristic interval sampling can select the
policy for M12. Consult `dispatch_changed` for each row.

These measurements do not establish gains for every eligible shape or for
other Hopper devices. H20, packed-K, group-scaled activations, other output
formats and large prefill grids are outside this policy. End-to-end performance
and composition with other tuning PRs remain unmeasured.

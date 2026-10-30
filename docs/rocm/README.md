# Humming on ROCm (AMD Instinct MI355X / gfx950)

Humming runs on ROCm through the same public API as on CUDA (`HummingLayer`, `humming.ops`,
`humming_forward`). Nothing changes for NVIDIA builds: every ROCm code path is selected at
runtime by `current_device.is_rocm` (Python) or at compile time by `HUMMING_HIP` (C++).

## Requirements

- AMD Instinct MI355X / MI350X (gfx950). Other GCN architectures are rejected at layer build.
- A ROCm build of PyTorch (tested: torch 2.11.0+rocm7.2) and a ROCm install for `hipcc`
  (tested: ROCm 7.1). The kernels load through the `libamdhip64` that ships with torch, so torch
  and Humming share one HIP runtime. ROCm 7.1 hipcc miscompiles some runtime selects at -O2+;
  `HUMMING_HIPCC_OPT_LEVEL=1` is a triage switch.

## Design

| layer | CUDA | ROCm |
|---|---|---|
| device-side primitives | `utils/ptx/*.cuh` (inline PTX) | `utils/hip/*.cuh`, selected only in `utils/platform.cuh` |
| JIT compiler | NVRTC / NVCC | `HIPCCCompiler` (`jit/compiler.py`): `-x hip --cuda-device-only --offload-arch=gfx950` |
| module load / launch | `cuda.bindings` driver API | `jit/hip.py` (ctypes over `libamdhip64`), via `KernelRuntime.launch` |
| C++ launcher | `<cuda.h>` | `csrc/launcher/gpu_api.h` maps the driver calls it uses onto HIP |
| GEMM kernels | `humming.cuh` / `humming_ws.cuh` / `humming_umma.cuh` | `humming_mfma.cuh` (fp16/bf16 activations), `humming_mfma_f8.cuh` (FP8 / MXFP4 / INT8 activations), `humming_mfma_moe.cuh` (MoE) |
| tile selection | `tune/sm*.py` | `tune/gfx950.py` |
| weight layout | Marlin-style repack | same repack, then `rocm/layout.py` rearranges it for the MFMA fragments (`transform.py`) |

`platform.cuh` is the only header that branches on `HUMMING_HIP`; the CUDA side of it is
exactly `#include <cuda.h>`. The process-input, quant / pack / repack and dequant kernels are
shared with CUDA and compile unchanged under hipcc apart from a few `HUMMING_HIP` blocks
(named barriers, FP8 / E2M1 / INT8 conversion helpers that replace PTX `cvt`).

The MFMA kernels use `v_mfma_f32_16x16x32_{f16,bf16}`, `v_mfma_scale_f32_16x16x128_f8f6f4`
(FP8 / FP6 / FP4 with E8M0 block scales in hardware) and `v_mfma_i32_16x16x64_i8`, wave64, and
load A straight into LDS with `global_load_lds`. Fragment layouts are documented in
`mfma_layout.md`; the tuning history is in `log_mfma.md`, `log_perf.md`, `log_f8.md`,
`log_moe.md` and `log_platform.md`.

## Support matrix

| activation | weight | notes |
|---|---|---|
| fp16 / bf16 | INT1-8 (symmetric or zero point), channel / group scales | dense + MoE |
| FP8 e4m3 / e5m2 | FP8, FP6, FP4, INT1-5 | per-tensor / channel / group / 128x128 block / MX scales |
| INT8 | INT1-8, MXFP4 | dense |
| MXFP4 | MXFP4 / MXFP6 / MXFP8, INT | dense |

MoE supports the indexed (vLLM), grouped-contiguous and grouped-masked layouts, w13 + w2,
per-expert bias, split-K and CUDA-graph capture, plus a HIP `moe_align_block_size`.

Not supported yet (raises `NotImplementedError`): NVFP4, FP / MX weights with fp16 / bf16
activations, MoE with low-bit activations, architectures other than gfx950.

## Performance (MI355X, single GPU)

Geomean over 10 Llama-3 / Qwen-style shapes, speedup vs the fastest existing ROCm quantized GEMM
for the same format (vLLM Triton W4A16, ExLlama GPTQ, AITER, dequant + hipBLASLt):

| config | M=1 | M=16 | M=64 | M=256 | M=1024 | M=4096 |
|---|---|---|---|---|---|---|
| fp16 W4A16 g128 | 4.13x | 4.20x | 2.98x | 1.84x | 1.19x | 0.95x |
| bf16 W4A16 g128 | 4.31x | 4.19x | 3.05x | 1.78x | 1.12x | 0.82x |
| bf16 W8A16 + zp | 3.80x | 3.58x | 3.02x | 1.67x | 1.02x | 0.74x |

Low-bit activations (llama3-70b down_proj, vs the best ROCm kernel for the same format):
W4A8 (FP8 x INT4 or MXFP4) 1.1-4.4x at every M; FP8 g128 x FP8 block-128 1.0-2.3x;
FP8 per-channel 0.57-1.0x; MXFP4 x MXFP4 0.26-1.28x.

MoE W4A16 (full block: align + w13 + SiLU-mul + w2 + top-k sum), Qwen3-30B-A3B /
Mixtral-8x7B / DeepSeek-V3, 1 to 8192 tokens: bf16 1.2-2.25x vs AITER a16wi4,
fp16 1.9-4.2x vs tuned vLLM Triton wna16.

Known gaps: at M >= 1024 the dense kernel is compute-bound and the weight dequant costs
17-19% over hipBLASLt fp16; FP8 per-channel and MXFP4 x MXFP4 lag AITER at large M.

## Tests

```
pytest tests/rocm
```

`tests/rocm` covers the platform layer, MFMA fragment layouts, every weight / activation
combination above against a torch reference, MoE layouts, and `HummingLayer` end to end.
The existing CUDA tests skip the cases that are gated on NVIDIA SM capabilities.

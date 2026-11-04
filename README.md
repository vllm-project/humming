
# Humming

Humming is a high-performance, lightweight, and highly flexible JIT (Just-In-Time) compiled GEMM kernel library specifically designed for quantized inference. It is developed by Ant Group and the vLLM community, and ships as the quantized GEMM backend in [vLLM](https://github.com/vllm-project/vllm).

Read the launch blog: [Announcing Humming: Fast, Flexible Kernels for Quantized Inference](https://vllm.ai/blog/2026-10-21-humming).


## Key Features

- **High Flexibility**
    - Supports inference for any weight type under 8-bit across **FP16 / BF16 / FP8 / FP4 / INT8 / INT4** activations (provided the activation's dynamic range covers the weight type).
    - Supports various quantization strategies (WNA16, W8A8, W4A8, W4A4, MXFP4/MXFP8, NVFP4, and more).
    - Supports various scale types (BF16, FP16, E4M3, E5M2, and UE8M0).
    - Supports both **Dense GEMM** and **MoE GEMM** (Indexed and Grouped layouts).
- **Broad Hardware Support**: every NVIDIA architecture from **SM75** (Turing) onward, including Hopper, Blackwell (SM100/103/110/120/121), and Rubin, plus CUDA-compatible devices such as T-Head's **Zhenwu 810E**.
- **High Performance**: outperforms strong baselines (vLLM Marlin, FlashInfer, DeepGEMM) across many representative dense and MoE workloads. See [Performance](#performance).
- **Ultra-Lightweight**
    * Minimal dependencies: Requires only **PyTorch** and a CUDA compiler (NVCC or NVRTC).
    * Compact footprint: the wheel is only about **500 KB**; kernels are compiled on demand and cached, typically in a few seconds per layer.


## Performance

Kernel microbenchmarks span M = 1 to 8192 with a cold L2 cache; MoE numbers sum the Gateup and Down GEMMs. Highlights:

| Workload | Device | Result |
| :--- | :--- | :--- |
| Dense W4A16 | Hopper / Blackwell | ~3–4× over cuBLAS at small M; up to ~2× vLLM Marlin on H200 and >3× on B300 at large M |
| MoE W4A16 (MXFP4 / MXINT4 / NVFP4) | H200 / B300 | ~2× vLLM Marlin on H200; up to ~5× on B300 (Grouped) |
| MoE W8A8 (FP8 block) | H20 / H200 | ~2.6× / ~2.5× DeepGEMM Contiguous at small–medium M |
| MoE W4A8 (MXFP4 × FP8) | H20 / H200 | ~4.2× / ~4.5× DeepGEMM W8A8, enabling fast MXFP4 on Hopper |
| MoE MX W8A8 / W4A8 / W4A4 | B300 | ~1.25× / 1.6× / 1.4× DeepGEMM at small–medium M |
| MoE W4A8 / W4A4 | SM120 / SM121 | Ahead of FlashInfer CUTLASS at most tested points |
| End-to-end DeepSeek-V4.1-Flash, 8×H20 | H20 | Humming W4A8 vs. Marlin W4A16: **+14–43%** throughput, **−25–32%** TTFT |

Full charts, setups, and software versions are in the [launch blog](https://vllm.ai/blog/2026-10-21-humming#performance).


## Support Matrix

| Activation Type | Supported Devices | Supported Weight Types |
| :--- | :--- | :--- |
| **FP16** (e5m10) | SM75+ | • Symmetric INT1-8<br>• INT1-8 with dynamic zero point<br>• Arbitrary signed FP (kBits ≤ 8, kExp ≤ 5) |
| **BF16** (e8m7) | SM80+ | • Symmetric INT1-8<br>• INT1-8 with dynamic zero point<br>• Arbitrary signed FP (kBits ≤ 8) |
| **FP8** (e4m3) | SM89+ | • Symmetric INT1-5<br>• INT1-4 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 4, kMan ≤ 3) |
| **FP8** (e5m2) | SM89+ | • Symmetric INT1-4<br>• INT1-3 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 5, kMan ≤ 2) |
| **FP4** (e2m1) | SM100+ | • Symmetric INT1-3<br>• INT1-2 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 2, kMan ≤ 1) |
| **INT8** | SM75+ | • Symmetric INT1-8<br>• INT1-7 with dynamic zero point |
| **INT4** | SM80+ | • Symmetric INT1-4<br>• INT1-3 with dynamic zero point |

On Blackwell (SM100/103/120/121), Humming also uses the hidden native `float4e0m3` and `float8e3m4` Tensor Core formats. SM103 (B300) has no INT8 UMMA, so INT8 there runs on the slower MMA path. The blog has a per-device chart of the [high-performance activation types](https://vllm.ai/blog/2026-10-21-humming#broad-device-support).


## Getting Started


### Installation

```bash
pip install humming-kernels
```

If your system has no CUDA compiler, pick the extra matching your PyTorch CUDA version:

```bash
pip install "humming-kernels[cu12]"  # CUDA 12
pip install "humming-kernels[cu13]"  # CUDA 13
```

Or install from source:

```bash
pip install git+https://github.com/vllm-project/humming.git
```


### Usage Example


```python
import torch
from humming.layer import HummingLayer

layer = HummingLayer(
    shape_n=8192,
    shape_k=8192,
    weight_config={"dtype": "int6"},
    torch_dtype=torch.float16,
).cuda()

weight = torch.randn((8192, 8192), dtype=torch.float16, device="cuda:0")
inputs = torch.randn((128, 8192), dtype=torch.float16, device="cuda:0")

# Load unquantized weight and quantize to layer quantization format
layer.load_from_unquantized(weight)
# Transform weight to humming format and prepare default kernels
layer.transform()

# Run quantized GEMM (tuning_config is optional, auto-selected by default)
output = layer(inputs)

print("Quantized GEMM Output:")
print(output)
print("\nReference Output:")
print(inputs.matmul(weight.T))
```

See [docs/config.md](docs/config.md) for the layer, compute, and tuning configuration reference.


## Acknowledgement

This project is highly inspired by

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/)
- [Marlin Kernel](https://github.com/IST-DASLab/marlin/) and [vLLM](https://github.com/vllm-project/vllm) Marlin Kernel
- [lmdeploy](https://github.com/InternLM/lmdeploy/) GEMM kernel
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [CUTLASS](https://github.com/nvidia/cutlass)

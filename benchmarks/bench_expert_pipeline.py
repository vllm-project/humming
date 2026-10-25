"""Opt-in SM90 grouped FC1 -> SwiGLU -> FC2 expert-band pipeline.

Compare whole, serial-banded, and concurrent-banded CUDA graphs with identical
weights and routing. Timings exclude routing, scatter/unpermute, and communication.
This is an experiment, not an automatic replacement for the default MoE path.
"""

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from humming.config import ComputeConfig, GemmType, LayerConfig
from humming.kernel.humming import HummingKernel
from humming.ops import launch_kernel
from humming.testing import KernelTestCase, KernelTestRunner


@triton.jit
def activate_band(
    up,
    down,
    offsets,
    START: tl.constexpr,
    END: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    first = tl.load(offsets + START)
    last = tl.load(offsets + END)
    if (row >= first) & (row < last):
        col = tl.arange(0, BLOCK)
        gate = tl.load(up + row * (2 * WIDTH) + col, col < WIDTH, other=0).to(tl.float32)
        value = tl.load(up + row * (2 * WIDTH) + WIDTH + col, col < WIDTH, other=0).to(tl.float32)
        gate = tl.minimum(gate, 10.0)
        value = tl.minimum(tl.maximum(value, -10.0), 10.0)
        result = gate / (1.0 + tl.exp(-gate)) * value
        tl.store(down + row * WIDTH + col, result, col < WIDTH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    parser.add_argument("--experts", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--bands", type=int, default=2)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--skew", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.experts or not 2 <= args.bands <= args.experts:
        parser.error("require 1 <= top-k <= experts and 2 <= bands <= experts")
    if min(args.tokens) < 1 or args.samples < 2:
        parser.error("tokens must be positive and samples must be at least two")
    if min(args.hidden, args.intermediate) <= 0 or args.hidden % 512 or args.intermediate % 256:
        parser.error("this tactic requires hidden multiple of 512 and intermediate multiple of 256")
    if torch.cuda.get_device_capability() != (9, 0):
        parser.error("this benchmark tactic is for SM90")

    torch.manual_seed(args.seed)
    compute = ComputeConfig(gemm_type=GemmType.GROUPED_CONTIGUOUS)
    # Fixed tactic across all arms; banding does not change arithmetic or tiling.
    tuning = dict(
        mma_type="wgmma",
        block_shape=(8, 512, 64),
        warp_shape=(8, 64, 64),
        num_stages=3,
        num_ctas_per_sm=3,
        use_stream_k=False,
        use_tma=True,
        use_tma_a=False,
        use_tma_b=True,
        use_tma_bs=True,
        use_tma_c=False,
    )
    bands = [
        (i * args.experts // args.bands, (i + 1) * args.experts // args.bands) for i in range(args.bands)
    ]
    whole = (0, args.experts)
    runners, kernels = {}, {}
    for name, n, k in (("fc1", 2 * args.intermediate, args.hidden), ("fc2", args.hidden, args.intermediate)):
        layer = LayerConfig(
            shape_n=n,
            shape_k=k,
            num_experts=args.experts,
            a_dtype="bfloat16",
            b_dtype="float4e2m1",
            c_dtype="bfloat16",
            bs_dtype="float8e8m0",
            weight_scale_group_size=32,
        )
        runners[name] = KernelTestRunner(
            KernelTestCase(
                name=name,
                layer_config=layer,
                compute_config=compute,
                seed=args.seed + (name == "fc2"),
                weight_std_scale=0.5,
            )
        )
        for band in [whole, *bands]:
            # The whole reference uses the unchanged default schedule.
            config = (
                compute
                if band == whole
                else ComputeConfig(gemm_type=compute.gemm_type, expert_start=band[0], expert_end=band[1])
            )
            kernels[name, band] = HummingKernel.prepare_kernels(
                layer.to_str(), config.to_str(), tuning, device=torch.device("cuda", 0)
            )

    streams = [torch.cuda.Stream() for _ in bands]
    for tokens in args.tokens:
        rows = tokens * args.top_k
        scores = torch.rand(tokens, args.experts, device="cuda")
        if args.skew:
            scores[: tokens * 3 // 4, : args.top_k] += 2
        ids = scores.topk(args.top_k, dim=1).indices.flatten()
        counts = torch.bincount(ids, minlength=args.experts)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))).int()
        inputs = torch.randn(rows, args.hidden, device="cuda", dtype=torch.bfloat16) * 0.5
        buffers = {
            arm: [
                torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
                for width in (2 * args.intermediate, args.intermediate, args.hidden)
            ]
            for arm in ("whole", "serial_bands", "parallel_bands")
        }

        def run(arm, buffers=buffers, inputs=inputs, offsets=offsets, rows=rows):
            up, activated, output = buffers[arm]
            parent = torch.cuda.current_stream()
            for index, band in enumerate([whole] if arm == "whole" else bands):
                stream = streams[index] if arm == "parallel_bands" else parent
                if arm == "parallel_bands":
                    stream.wait_stream(parent)
                with torch.cuda.stream(stream):
                    launch_kernel(
                        configs=kernels["fc1", band],
                        inputs=inputs,
                        outputs=up,
                        expert_layout=offsets,
                        valid_shape_m=rows,
                        **runners["fc1"].kernel_tensors,
                    )
                    activate_band[(rows,)](
                        up,
                        activated,
                        offsets,
                        *band,
                        args.intermediate,
                        triton.next_power_of_2(args.intermediate),
                    )
                    launch_kernel(
                        configs=kernels["fc2", band],
                        inputs=activated,
                        outputs=output,
                        expert_layout=offsets,
                        valid_shape_m=rows,
                        **runners["fc2"].kernel_tensors,
                    )
            if arm == "parallel_bands":
                for stream in streams:
                    parent.wait_stream(stream)

        graphs = {}
        for arm in buffers:
            run(arm)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(arm)
            graphs[arm] = graph
            graph.replay()
        for arm in ("serial_bands", "parallel_bands"):
            for actual, expected in zip(buffers[arm], buffers["whole"], strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        # Alternate order to reduce systematic clock/cache drift between arms.
        samples = {arm: [] for arm in graphs}
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for iteration in range(args.samples):
            order = list(graphs) if iteration % 2 == 0 else list(reversed(graphs))
            for arm in order:
                graphs[arm].replay()
                start.record()
                for _ in range(10):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end) * 100)
        medians = {arm: statistics.median(values) for arm, values in samples.items()}
        print(
            json.dumps(
                dict(
                    tokens=tokens,
                    experts=args.experts,
                    bands=args.bands,
                    skew=args.skew,
                    median_us=medians,
                    samples_us=samples,
                    exact_match=True,
                    latency_reduction_pct=100 * (1 - medians["parallel_bands"] / medians["whole"]),
                    scope="grouped FC1 + clamped SwiGLU + FC2; excludes routing and communication",
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

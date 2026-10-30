"""Benchmark + large-shape correctness for the gfx950 MFMA weight-only GEMM.

    HIP_VISIBLE_DEVICES=4 python -m tests.rocm.bench_mfma [--bits 4] [--group 128] [--zp] [--dtype fp16]
        [--check] [--ms 1,16,...] [--shapes 4096x4096,...]

Timing: HIP events around `iters` graph-captured launches after `warmup` launches, best of 5. Weight buffers
are rotated across enough copies to exceed the 256 MB MALL so the numbers reflect HBM.
Bytes = quantized weight + scales/zero points + activation + output.
"""

import argparse
import math

import torch

from tests.rocm.mfma_gemm import MfmaTile, mfma_gemm, select_tile
from tests.rocm.quant_utils import error_stats, make_quantized_linear

DEFAULT_MS = [1, 7, 16, 64, 255, 1024]
DEFAULT_SHAPES = [(4096, 4096), (11008, 4096), (4096, 11008), (8192, 28672), (28672, 8192)]
ROTATION_BYTES = 1 << 30


def time_rotating(fn, num_buffers, warmup=10, iters=100, repeats=5):
    """Average GPU time per call; calls are captured into a HIP graph so host launch
    overhead (ctypes + Python) does not leak into the measurement."""
    for i in range(warmup):
        fn(i % num_buffers)
    torch.cuda.synchronize()
    calls_per_graph = max(num_buffers, 20)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for i in range(calls_per_graph):
                fn(i % num_buffers)
    replays = max(1, iters // calls_per_graph)
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # The node is shared (other GPUs run production jobs): take the best of a few windows.
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0 / (replays * calls_per_graph))
    return best


def num_copies(nbytes):
    return max(1, min(64, math.ceil(ROTATION_BYTES / max(nbytes, 1))))


def bench_shape(shape_m, shape_n, shape_k, args, linear, dtype, tile=None, num_splits=None):
    weight_bytes = linear.mfma_weight.numel() * 4 + linear.mfma_scale.numel() * 2
    if linear.mfma_zero_point is not None:
        weight_bytes += linear.mfma_zero_point.numel()
    copies = num_copies(weight_bytes)
    weights = [linear.mfma_weight.clone() for _ in range(copies)]
    inputs = torch.randn(shape_m, shape_k, dtype=dtype, device="cuda")
    outputs = torch.empty(shape_m, shape_n, dtype=dtype, device="cuda")

    def run(i):
        mfma_gemm(
            inputs,
            weights[i],
            linear.mfma_scale,
            linear.mfma_zero_point,
            args.bits,
            args.group,
            tile=tile,
            num_splits=num_splits,
            outputs=outputs,
        )

    us = time_rotating(run, copies, args.warmup, args.iters)
    weights.clear()

    fp16_bytes = shape_n * shape_k * 2
    fp16_copies = num_copies(fp16_bytes)
    dense = linear.dequantized().to(dtype)
    dense_weights = [dense.clone() for _ in range(fp16_copies)]
    del dense
    us_ref = time_rotating(
        lambda i: torch.matmul(inputs, dense_weights[i].T, out=outputs), fp16_copies, args.warmup, args.iters
    )
    dense_weights.clear()

    io_bytes = shape_m * shape_k * 2 + shape_m * shape_n * 2
    flops = 2.0 * shape_m * shape_n * shape_k
    return {
        "us": us,
        "tbs": (weight_bytes + io_bytes) / us / 1e6,
        "tflops": flops / us / 1e6,
        "us_ref": us_ref,
        "tbs_ref": (fp16_bytes + io_bytes) / us_ref / 1e6,
        "speedup": us_ref / us,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--zp", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--ms", default=",".join(map(str, DEFAULT_MS)))
    parser.add_argument("--shapes", default=",".join(f"{n}x{k}" for n, k in DEFAULT_SHAPES))
    parser.add_argument("--check", action="store_true", help="also check correctness vs FP32 torch")
    parser.add_argument("--check-only", action="store_true", help="correctness only, no timing")
    parser.add_argument(
        "--tile", default=None, help="warps_m,warps_n,tiles_m,tiles_n,stages[,lds_a[,scale_b]]"
    )
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    args.check = args.check or args.check_only

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    ms = [int(m) for m in args.ms.split(",")]
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]
    tile = None
    if args.tile:
        fields = [int(x) for x in args.tile.split(",")]
        fields += [0] * (7 - len(fields))
        tile = MfmaTile(*fields[:5], lds_a=bool(fields[5]), scale_b=bool(fields[6]))

    print(f"# bits={args.bits} group={args.group} zp={args.zp} dtype={args.dtype}")
    header = "| M | N | K | tile (wm,wn,tm,tn,st) x splits |"
    if not args.check_only:
        header += " us | TB/s | TFLOPs | torch us | torch TB/s | speedup |"
    if args.check:
        header += " max abs err | max rel err |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for shape_n, shape_k in shapes:
        linear = make_quantized_linear(shape_n, shape_k, args.bits, args.group, args.zp, dtype)
        for shape_m in ms:
            selected_tile, selected_splits = select_tile(
                shape_m, shape_n, shape_k, args.bits, is_bfloat16=dtype == torch.bfloat16
            )
            used_tile = tile or selected_tile
            used_splits = args.splits or selected_splits
            t = used_tile
            tile_str = f"{t.warps_m},{t.warps_n},{t.tiles_m},{t.tiles_n},{t.stages}"
            tile_str += (",lds" if t.lds_a else "") + (",sb" if t.scale_b else "")
            row = f"| {shape_m} | {shape_n} | {shape_k} | {tile_str} x{used_splits} |"
            if not args.check_only:
                r = bench_shape(shape_m, shape_n, shape_k, args, linear, dtype, tile, args.splits)
                row += f" {r['us']:.1f} | {r['tbs']:.2f} | {r['tflops']:.1f} |"
                row += f" {r['us_ref']:.1f} | {r['tbs_ref']:.2f} | {r['speedup']:.2f}x |"
            if args.check:
                inputs = torch.randn(shape_m, shape_k, dtype=dtype, device="cuda")
                outputs = mfma_gemm(
                    inputs,
                    linear.mfma_weight,
                    linear.mfma_scale,
                    linear.mfma_zero_point,
                    args.bits,
                    args.group,
                    tile=tile,
                    num_splits=args.splits,
                )
                reference = inputs.float() @ linear.dequantized().T
                max_abs, max_rel = error_stats(outputs, reference)
                row += f" {max_abs:.3e} | {max_rel:.2e} |"
            print(row, flush=True)
        del linear
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

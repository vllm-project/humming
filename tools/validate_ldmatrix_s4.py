"""Direct one-GPU validation; run with HUMMING_COMPILER=nvcc."""

import argparse
import functools
import json
import os
import statistics
import subprocess
from pathlib import Path


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_report(output_path, manifest, reports):
    import torch

    query = ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]
    query += ["--id", manifest["gpu_uuid"]]
    gpu, driver, capability = [
        part.strip() for part in subprocess.check_output(query, text=True).strip().split(",")
    ]
    environment = dict(
        gpu_name=gpu,
        driver_version=driver,
        compute_capability=capability,
        cuda_runtime_version=torch.version.cuda,
        nvcc_version=subprocess.check_output(["nvcc", "--version"], text=True).strip(),
        compute_sanitizer_version=subprocess.check_output(
            ["compute-sanitizer", "--version"], text=True
        ).strip(),
    )
    result = dict(
        schema_version=1,
        production_sha=revision(),
        baseline_sha=manifest["baseline_sha"],
        status="passed",
        executed_cases=len(reports),
        skipped_cases=0,
        cases=reports,
        environment=environment,
    )
    torch.cuda.synchronize()
    with output_path.open("x") as stream:
        json.dump(result, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader-mode", choices=("automatic", "force_legacy", "both"), default="both")
    parser.add_argument("--shape", choices=("model-real",), default="model-real")
    parser.add_argument("--scale-pattern", choices=("nonuniform",), default="nonuniform")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--gate",
        choices=("selection", "identity", "parity", "stability", "memcheck", "benchmark"),
        default="memcheck",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.output_json and not args.manifest:
        parser.error("--output-json requires --manifest with frozen provenance")
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    if manifest and manifest["production_sha"] != revision():
        raise RuntimeError("production SHA differs from frozen manifest")
    if manifest:
        subprocess.run(["git", "cat-file", "-e", manifest["baseline_sha"] + "^{commit}"], check=True)
    if args.warmup < 1 or args.samples < 1:
        parser.error("warmup and samples must be positive")
    import torch

    from humming.testing.ldmatrix_s4 import (
        PreparedDense,
        assert_output,
        compare_outputs,
        input_data,
        make_config,
        weight_data,
    )
    from humming.tune import get_heuristics_config

    if os.environ.get("HUMMING_COMPILER") != "nvcc":
        raise RuntimeError("set HUMMING_COMPILER=nvcc for this validated toolchain contract")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this runner requires one SM90 GPU")
    if not make_config().can_use_ldmatrix_s4:
        raise RuntimeError(str(make_config().ldmatrix_s4_rejection_reasons))
    torch.backends.cuda.matmul.allow_tf32 = False
    print(
        json.dumps(
            dict(
                revision=revision(),
                gpu=torch.cuda.get_device_name(),
                torch_cuda=torch.version.cuda,
                compiler=subprocess.check_output(["nvcc", "--version"], text=True),
                driver=subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
                ).strip(),
                cache=os.environ.get("HUMMING_CACHE_DIR", "default"),
                warmup=args.warmup,
                samples=args.samples,
            )
        ),
        flush=True,
    )
    if args.gate in ("selection", "identity", "parity", "stability"):
        import pytest

        from humming.testing.ldmatrix_s4 import VALIDATION_RESULTS

        selections = {
            "selection": (
                "selection_metadata or ineligible_layer or ineligible_toolchain or moe_excluded "
                "or incompatible_specialization or wrong_orientation or graph_capture or misaligned_repacked "
                "or native_moe_routing"
            ),
            "identity": "layout_cache_and_mode_switching or force_legacy_does_not_mutate",
            "parity": "dense_production_parity or nonuniform_scale_oracle",
            "stability": "compact_batch_and_alternation",
        }

        class ExecutionStatus:
            skipped = 0

            def pytest_runtest_logreport(self, report):
                if report.skipped:
                    self.skipped += 1

        execution = ExecutionStatus()
        VALIDATION_RESULTS.clear()
        code = pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/kernels/humming/test_ldmatrix_s4_dense.py",
                "-k",
                selections[args.gate],
            ],
            plugins=[execution],
        )
        if code or execution.skipped or not VALIDATION_RESULTS:
            raise RuntimeError(f"{args.gate} did not complete all required cases (pytest={code})")
        if args.output_json:
            write_report(args.output_json, manifest, VALIDATION_RESULTS)
        else:
            print(json.dumps(VALIDATION_RESULTS), flush=True)
        return
    args.benchmark = args.benchmark or args.gate == "benchmark"
    shapes = [(16, 4096, 4096)]
    if args.benchmark:
        shapes = [(1, 4096, 4096), (32, 4096, 4096), (256, 4096, 4096), (128, 128, 128)]
    reports = []
    for m, n, k in shapes:
        data, reference = weight_data(n, k, random=True, nonuniform=True)
        inputs = [input_data(m, k, seed=seed) for seed in (0, 1)]
        new_config = make_config(n, k)
        tuning = get_heuristics_config(new_config, shape_m=m, gemm_type="dense")
        modes = (False, True) if args.loader_mode == "both" else (args.loader_mode == "force_legacy",)
        legacy_prepared = PreparedDense.create(make_config(n, k, legacy=True), data, reference)
        legacy_outputs = [legacy_prepared.run(value, tuning=tuning)[0] for value in inputs]
        for legacy in modes:
            prepared = PreparedDense.create(make_config(n, k, legacy=legacy), data, reference)
            for index in range(12):
                value = inputs[index % 2]
                output, kernel = prepared.run(value, tuning=tuning)
                report = compare_outputs(
                    output,
                    legacy_outputs[index % 2],
                    prepared.oracle(value),
                    kernel,
                    m=m,
                    n=n,
                    k=k,
                    scale_pattern="nonuniform",
                )
            torch.cuda.synchronize()
            mode = "force_legacy" if legacy else "automatic"
            report.update(
                case_id=f"m{m}-n{n}-k{k}-{mode}",
                tuning=tuning,
                iterations=12,
                warmup=args.warmup,
                measurement_count=args.samples,
                cache_state="warm_same_tuning",
                cache_root=os.environ.get("HUMMING_CACHE_DIR", "default"),
            )
            if args.benchmark:
                # Preallocate launch inputs and time only the registered kernel path.
                from humming import ops

                output_buffer = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                input_scale = torch.full((m, 1), 1 / 128, device="cuda")
                configs = torch.tensor([kernel.kernel_id], dtype=torch.int64)

                launch = functools.partial(
                    ops.launch_kernel,
                    configs=configs,
                    inputs=inputs[0],
                    **prepared.tensors,
                    input_scale=input_scale,
                    outputs=output_buffer,
                )

                for _ in range(args.warmup):
                    launch()
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.samples):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    launch()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                assert_output(output_buffer, prepared.oracle(inputs[0]), kernel)
                report.update(median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples))
            reports.append(report)
            print(json.dumps(report), flush=True)
    if args.output_json:
        write_report(args.output_json, manifest, reports)


if __name__ == "__main__":
    main()

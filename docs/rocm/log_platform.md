# ROCm platform port log (branch `rocm-platform`)

Scope: device detection, HIP JIT compiler, runtime/launch, device-header portability
and the non-MMA kernels (pack/unpack/quant/dequant/repack weight, process_mxfp4,
process_input) on MI355X (gfx950). The MFMA GEMM is out of scope.

Environment: ROCm 7.1 at `/opt/rocm`, torch 2.11.0+rocm7.2 in
`/home/zhenchen/projects/humming-rocm/.venv-rocm`, GPU 3 only (`HIP_VISIBLE_DEVICES=3`).

## Step 1: device detection

Changes:
- `humming/utils/rocm.py` (new): `is_rocm_build()` (from `torch.version.hip`), ROCm
  path/version/hipcc discovery, `normalize_gcn_arch`.
- `humming/csrc/device_info.cpp`: `#ifdef HUMMING_HIP` branch using the HIP runtime
  (`hipGetDeviceProperties` + `hipDeviceGetAttribute`). New attributes on both
  platforms: `is_rocm`, `arch` (`sm_XY` on CUDA, `gfx950` on ROCm), `warp_size`.
  The CUDA repr is unchanged; ROCm appends the three new fields.
- `humming/utils/device.py`: ROCm build (`g++ -DHUMMING_HIP=1 -D__HIP_PLATFORM_AMD__
  -I/opt/rocm/include -lamdhip64`), precompiled CUDA artifact ignored on ROCm, ROCm
  version folded into the build hash.
- `humming/device.py`: `is_rocm`, `arch`, `warp_size` properties.

Decisions:
- `sm_major = sm_minor = sm_version = 0` on ROCm. HIP reports gfx950 as compute
  capability 9.5, which every `sm_version >= 90` gate would read as Hopper
  (WGMMA/TMA/PDL/cluster). 0 makes all such gates take the oldest path; code that
  needs the AMD arch keys on `current_device.is_rocm` / `current_device.arch`.
- Memory bandwidth on CDNA uses 4 transfers per reported memory clock
  (2000 MHz x 8192 bit x 4 / 8 = 8.19 TB/s on MI355X, matches spec).
- `tensorcore_tops` on ROCm: dense MFMA rate table (gfx950: 4096 FP16 ops/CU/clk,
  FP8/INT8 x2, FP6/FP4 x4, no INT4/E3M4/E0M3).

Result (`HIP_VISIBLE_DEVICES=3 python -c "from humming.device import current_device; current_device.print()"`):
`name='AMD Instinct MI355X', sm_count=256, sm_version=0, l2=4 MB, l1=32 KB,
default_smem_size=max_smem_size=163840, sm_clock_khz=2400000,
memory_bandwidth_gbps=8192.0, float16 2516 TOPS, is_rocm=True, arch='gfx950', warp_size=64`.

## Step 2: hipcc JIT compiler

Changes (`bb13372`, `6a04bfe`): `HIPCCCompiler` in `humming/jit/compiler.py`, selected by
`KernelRuntime._get_compiler()` when `current_device.is_rocm`. `KernelRuntime.init_sm_version`
passes `current_device.arch` (`gfx950`) as the compiler target.

Flags: `-x hip --cuda-device-only --no-gpu-bundle-output --offload-arch=gfx950 -std=c++17
-O${HUMMING_HIPCC_OPT_LEVEL:-3} -DHUMMING_HIP=1 -I<humming/include> -Wno-unused-result
-Wno-unused-value -Wno-deprecated-declarations`, plus `-fgpu-flush-denormals-to-zero
-ffp-contract=fast -fapprox-func` as the counterpart of `--use_fast_math` (dropped when
`disable_fast_math`). The cache signature is `hipcc+rocm-<version>+<arch>`.

Failures and decisions:
- `--genco` alone writes a clang offload bundle (`__CLANG_OFFLOAD_BUNDLE__`). Neither
  `hipModuleLoad` nor the ELF reader take it, so `--cuda-device-only --no-gpu-bundle-output`
  emits a raw AMDGPU ELF (e_machine 0xE0). The cache file keeps the name `kernel.cubin`.
- Device-only compilation drops templates that are never instantiated, so `kernel_expr` is
  pinned with `auto ptr = reinterpret_cast<void*>(&<expr>);`, as on NVRTC.
- Kernel names come from the `<name>.kd` STT_OBJECT symbols (`humming/utils/cubin.py`).

## Step 3: runtime / launch

- `humming/jit/hip.py` is a ctypes binding of `hipModuleLoad`, `hipModuleGetFunction`,
  `hipModuleGetGlobal`, `hipFuncSetAttribute` and `hipModuleLaunchKernel`.
- `KernelRuntime.launch(func, grid, block, stream, arg_values, arg_types, shared_memory_bytes,
  use_pdl)` is the single launch helper. CUDA keeps using `cuLaunchKernelEx`; ROCm uses
  `hip.launch_kernel`, and `use_pdl` is ignored. All 7 weight kernels plus tops_bench go
  through it.

**Bug: duplicate HIP runtime (`c802493`).** `hip.py` loaded `/opt/rocm/lib/libamdhip64.so`.
The torch wheel ships its own `torch/lib/libamdhip64.so`, so the process held two
independent HIP runtimes and torch's stream handles meant nothing to the second one.
Symptoms:
- repack output was nondeterministic or never written;
- results were correct only after an extra synchronize.

Fix: `rocm_utils.get_amdhip_library_path()` initializes torch's CUDA, then picks the
`libamdhip64.so*` already mapped (scanning `/proc/self/maps`), then torch's copy, then ROCm's.
`tests/rocm/test_platform.py::test_single_hip_runtime_is_mapped` and
`test_launch_is_ordered_with_torch_stream` guard this.

## Step 4: device-header portability (`6a04bfe`)

- `humming/include/humming/utils/platform.cuh` is the only header that branches on
  `HUMMING_HIP`. See the contract comment at the top of that file.
- `humming/include/humming/utils/hip/*.cuh` replace the PTX helpers:
  - `math.cuh`: prmt via v_perm, lop3 via minterm expansion;
  - `barrier.cuh`: Stream-K locks via HIP atomics;
  - `warp.cuh`, `shared.cuh`;
  - error-on-use stubs for TMA, WGMMA, mbarrier, ldmatrix and cp.async (`tma.cuh`,
    `wgmma.cuh`, `unsupported.cuh`).
- Logical warps are 32 lanes inside a 64-lane wavefront. Shuffles, reductions and ballot act
  on the logical warp. The mask is accepted and ignored.
- `HUMMING_NAMED_BARRIER_SYNC(id, n)` falls back to `__syncthreads()`. This is valid only when
  every thread runs the same barrier sequence.

## Step 5: non-MMA kernels

**hipcc scratch miscompile (`efb7eb8`).** On ROCm 7.1 at -O2/-O3, `weight_repack_nk` lowers the
runtime `interleave_mode` selects into a negative SGPR scratch base
(`s_sub_i32 sN, 0, ...; scratch_load ... off, sN`). The result faults, or silently writes
garbage, on gfx950.
- It reproduces in a standalone C++ HIP program (`/tmp/hb/r48dump.hip`), so this is a compiler
  bug rather than a humming bug.
- At -O1 it is correct.
- Workaround: under `HUMMING_HIP`, dispatch to compile-time interleave modes (normalized to
  0..3, since `humming_pack_weight` only reads `mode % 2` and `mode / 2 == 0`).
- `HUMMING_HIPCC_OPT_LEVEL` was added as a debugging aid.

**process_input (`74d02ac`):**
- `quantization.cuh` gets HIP pack paths:
  - fp8 e4m3/e5m2 via `__builtin_amdgcn_cvt_pk_{fp8,bf8}_f32` (OCP on gfx950) after a ±max clamp
    that keeps NaN;
  - e2m1 via software RNE with satfinite and signed zero;
  - int8 via `rintf` plus a clamp to [-128,127], with NaN → 0;
  - int4 the same way.
- `hadamard.cuh` and `sync_slice_threads` use `HUMMING_NAMED_BARRIER_SYNC`. kWarpOffset is
  always 0 and the barriers are uniform, so the `__syncthreads()` fallback is valid there.
- `activation.cuh`: bf16 unpacking with shift/mask.
- `tune/process_input.py`: on ROCm the dtype gate is
  `{int4, int8, float8e4m3, float8e5m2, float4e2m1}`. E3M4/E0M3 rely on NVIDIA cubin
  patching and are unsupported.
- The tests skip E3M4/E0M3 on HIP, and `test_quantization_support_matches_cubin_patcher` is
  skipped on HIP.

**C++ launcher (`24d1949`):**
- `humming/csrc/launcher/gpu_api.h` is exactly `<cuda.h>` on CUDA. On HIP it maps the driver
  API subset:
  - contexts become per-device, with push/pop via a thread_local device stack;
  - `cuLaunchKernelEx` becomes `hipModuleLaunchKernel`; cluster > 1 is rejected and PDL is
    ignored;
  - `cuTensorMapEncodeTiled` returns NOT_SUPPORTED.
- `elf.h` reads AMDGPU symbol tables: `.kd` symbols give the names, and the file offset is
  `sh_offset + st_value - sh_addr`.
- `ops/utils.py` links torch's `libamdhip64`, `c10_hip` and `torch_hip`, and skips precompiled
  artifacts on ROCm.
- Check: `init_humming_launcher()` builds and loads on GPU 3.

## Step 6: HummingLayer up to the GEMM (`47f6d21`)

Fixes:
- `config._cuda_compiler_version` parses `rocm-*` as (0, 0).
- `layer.py`: bf16 is the default on ROCm.
- `forward._resolve_use_pdl` returns False on ROCm. It was hitting a KeyError on sm 0 in
  `get_heuristics_class`.

`HummingKernel._resolve_configs` raises
`NotImplementedError("MFMA GEMM backend not yet integrated")` on ROCm. That is the hook for the
MFMA backend.

Remaining NVIDIA `sm_version` gates, all on the GEMM side:
- `kernel/humming.py` check_dtype asserts (`sm_version >= dtype_map`) and the WGMMA/UMMA asserts;
- `schema/humming.py` `dtype_min_sm_version_map`;
- `tune/__init__.py` `get_heuristics_class`, which has no ROCm entry.

## Test results (GPU 3)

```
V=/home/zhenchen/projects/humming-rocm/.venv-rocm/bin/python
HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD $V -m pytest -q tests/rocm -p no:cacheprovider
  -> 66 passed
HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD $V -m pytest -q tests/kernels/process_input -p no:cacheprovider
  -> 276 passed, 24 skipped (E3M4/E0M3 + cubin-patcher capability test)
HIP_VISIBLE_DEVICES=3 PYTHONPATH=$PWD $V -m pytest -q tests/test_jit_infrastructure.py \
  tests/test_sm8x_heuristics.py tests/test_sm90_h20_heuristics.py tests/test_sm90_heuristics.py \
  tests/test_tuning_candidates.py -p no:cacheprovider
  -> 500 passed, 150 skipped, 5 failed
```

The 5 failures are pre-existing and device-independent. In `test_sm90_h20_heuristics` mxfp4,
`_mxfp4_layer` sets c_dtype=bf16 with a_dtype=fp16, and the schema rejects that
("16-bit a_dtype float16 requires matching c_dtype").

`tests/rocm`:
- `test_platform.py`:
  - device fields;
  - a single mapped libamdhip64;
  - a raw AMDGPU ELF with kernel names;
  - launch ordering on torch's stream with no explicit sync;
  - the 32-lane logical-warp `__shfl_sync`.
- `test_weight_kernels.py`:
  - pack/unpack bits 1-8, exact against a reference packer;
  - dequant e4m3/e5m2 against torch, and the e2m1 table;
  - the quant round trip (7 cases x 3 source dtypes);
  - process_mxfp4 against a Python reference (input -0 is canonicalized; outputs that round to
    zero keep the sign);
  - repack wb {2,4,8} x ab {16,8} x modes 0-3: deterministic, x not modified, packed input ==
    unpacked input, and the value histogram preserved for modes 0/3 and w8a8.
- `test_layer.py`: int4, int8, fp8e4m3 and mxfp4 (fp8 input) transform, then forward raises
  NotImplementedError.

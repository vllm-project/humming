import dataclasses
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import torch

from humming import dtypes
from humming.device import current_device
from humming.jit import hip as hip_runtime
from humming.jit.compiler import HIPCCCompiler, NVCCCompiler, NVRTCCompiler
from humming.utils.rocm import is_rocm_build

if is_rocm_build():
    cbd: Any = None
else:
    import cuda.bindings.driver as cbd
from humming.utils.cpu import get_parallel_build_workers
from humming.utils.cubin import get_cubin_kernel_names


@dataclasses.dataclass(kw_only=True)
class KernelRuntime:
    disable_fast_math: ClassVar[bool] = False
    _instances: ClassVar[dict[tuple[Any, ...], "KernelRuntime"]] = {}

    def __new__(cls, *args, **kwargs):
        def get_value(value):
            if isinstance(value, dtypes.DataType):
                return str(value)
            if isinstance(value, list):
                value = tuple(value)
            return value

        args_items = tuple(get_value(x) for x in args)
        kwargs_items = tuple((key, get_value(kwargs[key])) for key in sorted(kwargs.keys()))
        signature = (cls.__name__, current_device.index, args_items + kwargs_items)

        if signature not in cls._instances or not cls._instances[signature].inited:
            instance = super().__new__(cls)
            cls._instances[signature] = instance
            instance.inited = False
            instance._context_kernels = {}
            instance._kernel_load_lock = threading.Lock()
        return cls._instances[signature]

    def __post_init__(self):
        if self.inited:
            return
        self.init_sm_version()
        self.init_kernel()
        self.inited = True
        if not hasattr(self, "_vars"):
            self._vars = vars(self)
        else:
            for key, value in self._vars.items():
                setattr(self, key, value)

    def init_kernel(self):
        raise NotImplementedError

    def init_sm_version(self):
        self.sm_version = current_device.sm_version
        if current_device.is_rocm:
            # sm_version is 0 on ROCm; the compiler target is the GCN arch.
            self.sm_version_str = current_device.arch
            return
        self.sm_version_str = str(self.sm_version)
        if self.sm_version >= 90:
            self.sm_version_str += "a"

    @staticmethod
    def _get_compiler():
        if current_device.is_rocm:
            return HIPCCCompiler
        compiler = os.environ.get("HUMMING_COMPILER", "").lower()
        if compiler == "nvcc":
            return NVCCCompiler
        elif compiler == "nvrtc":
            return NVRTCCompiler
        elif current_device.is_ppu:
            return NVCCCompiler
        else:
            try:
                from cuda.bindings import nvrtc  # noqa

                return NVRTCCompiler
            except Exception:
                return NVCCCompiler

    @staticmethod
    def current_context():
        if current_device.is_rocm:
            # HIP modules are per device; the device index stands in for the context.
            return hip_runtime.get_device()
        result, context = cbd.cuCtxGetCurrent()
        assert result == 0, repr(result)
        return context

    def _ensure_cuda_context(self):
        if current_device.is_rocm:
            torch.cuda.current_device()  # initializes the HIP runtime through torch
            return self.current_context()
        context = self.current_context()
        if context == cbd.CUcontext():
            torch.cuda.set_device(torch.cuda.current_device())
            context = self.current_context()
        return context

    def prepare(self):
        compiler_cls = self._get_compiler()
        kernel_expr = getattr(self, "kernel_expr", None)
        kernel_filename = compiler_cls.compile(
            self.code,
            sm_version=self.sm_version_str,
            kernel_expr=kernel_expr,
            disable_fast_math=self.disable_fast_math,
            postprocess_cubin=self.postprocess_cubin,
        )
        self.kernel_filename = kernel_filename

    @staticmethod
    def compile_many(kernel_specs, device=None):
        device = torch.cuda.current_device() if device is None else device

        def compile_kernel(spec):
            kernel_type, kernel_args = spec
            with torch.cuda.device(device):
                return kernel_type(**kernel_args)

        kernel_specs = list(kernel_specs)
        workers = min(len(kernel_specs), get_parallel_build_workers()) if kernel_specs else 0
        parallel = workers > 1
        if parallel:
            with ThreadPoolExecutor(
                max_workers=workers,
            ) as executor:
                kernels = list(executor.map(compile_kernel, kernel_specs))
        else:
            kernels = [compile_kernel(spec) for spec in kernel_specs]
        return kernels

    def postprocess_cubin(self, cubin_path: str):
        pass

    def load_cubin(self):
        context = self._ensure_cuda_context()
        if context in self._context_kernels:
            return self._context_kernels[context][1]

        with self._kernel_load_lock:
            if context in self._context_kernels:
                return self._context_kernels[context][1]
            kernel_filename = self.kernel_filename
            if current_device.is_rocm:
                module = hip_runtime.module_load(kernel_filename)
                matched = [name for name in get_cubin_kernel_names(kernel_filename) if self.name in name]
                assert len(matched) == 1, (kernel_filename, self.name, matched)
                self.kernel_name = matched[0]
                func = hip_runtime.module_get_function(module, self.kernel_name)
                self._context_kernels[context] = (module, func)
                return func
            result, module = cbd.cuModuleLoad(kernel_filename.encode())
            assert result == 0, repr(result)
            matched = [name for name in get_cubin_kernel_names(kernel_filename) if self.name in name]
            assert len(matched) == 1, (kernel_filename, self.name, matched)
            self.kernel_name = matched[0]
            result, func = cbd.cuModuleGetFunction(module, self.kernel_name.encode())
            assert result == 0, repr(result)
            self._context_kernels[context] = (module, func)
            return func

    def launch(
        self,
        func: Any,
        grid: Sequence[int],
        block: Sequence[int],
        stream: int,
        arg_values: Sequence[Any],
        arg_types: Sequence[Any] | None = None,
        shared_memory_bytes: int = 0,
        use_pdl: bool = False,
    ) -> None:
        """Launch ``func`` (from ``load_cubin``) on ``stream``.

        ``arg_types`` are ctypes types (defaults to ``self.arg_types``). CUDA goes through
        ``cuLaunchKernelEx`` exactly as before; ROCm goes through ``hipModuleLaunchKernel``
        and ignores ``use_pdl`` (no programmatic dependent launch on AMD).
        """
        arg_types = self.arg_types if arg_types is None else arg_types
        grid_dims = (*grid, 1, 1)[:3]
        block_dims = (*block, 1, 1)[:3]
        if current_device.is_rocm:
            hip_runtime.launch_kernel(
                func, grid_dims, block_dims, stream, arg_values, arg_types, shared_memory_bytes
            )
            return
        config = cbd.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid_dims
        config.blockDimX, config.blockDimY, config.blockDimZ = block_dims
        config.sharedMemBytes = shared_memory_bytes
        config.hStream = stream
        self.set_pdl_launch_attribute(config, use_pdl)
        cbd.cuLaunchKernelEx(config, func, (tuple(arg_values), tuple(arg_types)), 0)

    def set_pdl_launch_attribute(self, config: Any, enabled: bool) -> None:
        if not enabled or self.sm_version < 90:
            return
        attr = cbd.CUlaunchAttribute()
        attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attr.value.programmaticStreamSerializationAllowed = 1
        config.attrs = [attr]
        config.numAttrs = 1

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

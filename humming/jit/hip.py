"""Minimal ctypes binding of the HIP module API used to load and launch JIT kernels on ROCm.

Only the handful of entry points humming needs are bound: module load, function lookup,
function attributes and ``hipModuleLaunchKernel``. Everything is resolved lazily so the
module can be imported on CUDA builds without libamdhip64 present.
"""

import ctypes
import functools
import threading
from collections.abc import Sequence
from typing import Any

from humming.utils.rocm import get_amdhip_library_path

HIP_SUCCESS = 0
# hipFuncAttribute_t::hipFuncAttributeMaxDynamicSharedMemorySize
HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE = 8
# Dynamic LDS above this needs the attribute raised first, as on CUDA.
DEFAULT_DYNAMIC_SHARED_MEMORY_LIMIT = 64 * 1024

HipModule = ctypes.c_void_p
HipFunction = ctypes.c_void_p


@functools.cache
def _library() -> ctypes.CDLL:
    library = ctypes.CDLL(get_amdhip_library_path(), mode=ctypes.RTLD_GLOBAL)
    library.hipGetErrorName.restype = ctypes.c_char_p
    library.hipGetErrorName.argtypes = [ctypes.c_int]
    library.hipGetErrorString.restype = ctypes.c_char_p
    library.hipGetErrorString.argtypes = [ctypes.c_int]
    library.hipGetDevice.restype = ctypes.c_int
    library.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.hipModuleLoad.restype = ctypes.c_int
    library.hipModuleLoad.argtypes = [ctypes.POINTER(HipModule), ctypes.c_char_p]
    library.hipModuleLoadData.restype = ctypes.c_int
    library.hipModuleLoadData.argtypes = [ctypes.POINTER(HipModule), ctypes.c_void_p]
    library.hipModuleUnload.restype = ctypes.c_int
    library.hipModuleUnload.argtypes = [HipModule]
    library.hipModuleGetFunction.restype = ctypes.c_int
    library.hipModuleGetFunction.argtypes = [ctypes.POINTER(HipFunction), HipModule, ctypes.c_char_p]
    library.hipModuleGetGlobal.restype = ctypes.c_int
    library.hipModuleGetGlobal.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        HipModule,
        ctypes.c_char_p,
    ]
    library.hipFuncSetAttribute.restype = ctypes.c_int
    library.hipFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.hipModuleLaunchKernel.restype = ctypes.c_int
    library.hipModuleLaunchKernel.argtypes = [
        HipFunction,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    return library


def check_hip(result: int, function_name: str) -> None:
    if result == HIP_SUCCESS:
        return
    library = _library()
    error_name = library.hipGetErrorName(result).decode()
    error_string = library.hipGetErrorString(result).decode()
    raise RuntimeError(f"{function_name} failed with HIP error {result}: {error_name} ({error_string})")


def get_device() -> int:
    device = ctypes.c_int()
    check_hip(_library().hipGetDevice(ctypes.byref(device)), "hipGetDevice")
    return device.value


def module_load(path: str) -> int:
    module = HipModule()
    check_hip(_library().hipModuleLoad(ctypes.byref(module), path.encode()), "hipModuleLoad")
    assert module.value is not None
    return module.value


def module_get_function(module: int, name: str) -> int:
    function = HipFunction()
    result = _library().hipModuleGetFunction(ctypes.byref(function), HipModule(module), name.encode())
    check_hip(result, f"hipModuleGetFunction({name})")
    assert function.value is not None
    return function.value


def module_get_global(module: int, name: str) -> tuple[int, int]:
    """Return ``(device_pointer, size_in_bytes)`` of a ``__device__``/``__constant__`` symbol."""
    pointer = ctypes.c_void_p()
    size = ctypes.c_size_t()
    result = _library().hipModuleGetGlobal(
        ctypes.byref(pointer), ctypes.byref(size), HipModule(module), name.encode()
    )
    check_hip(result, f"hipModuleGetGlobal({name})")
    return pointer.value or 0, size.value


_dynamic_smem_limits: dict[int, int] = {}
_dynamic_smem_lock = threading.Lock()


def ensure_dynamic_shared_memory(function: int, num_bytes: int) -> None:
    if num_bytes <= DEFAULT_DYNAMIC_SHARED_MEMORY_LIMIT:
        return
    with _dynamic_smem_lock:
        if _dynamic_smem_limits.get(function, 0) >= num_bytes:
            return
        result = _library().hipFuncSetAttribute(
            ctypes.c_void_p(function), HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE, num_bytes
        )
        check_hip(result, "hipFuncSetAttribute(MaxDynamicSharedMemorySize)")
        _dynamic_smem_limits[function] = num_bytes


def launch_kernel(
    function: int,
    grid: Sequence[int],
    block: Sequence[int],
    stream: int,
    arg_values: Sequence[Any],
    arg_types: Sequence[Any],
    shared_memory_bytes: int = 0,
) -> None:
    """Launch with the same ``(values, ctypes types)`` argument convention as cuLaunchKernelEx."""
    assert len(arg_values) == len(arg_types), (arg_values, arg_types)
    grid_x, grid_y, grid_z = (*grid, 1, 1)[:3]
    block_x, block_y, block_z = (*block, 1, 1)[:3]
    # Keep every boxed argument alive until hipModuleLaunchKernel has copied it.
    boxed_args = [
        arg_type(0 if value is None else value) for value, arg_type in zip(arg_values, arg_types, strict=True)
    ]
    arg_pointers = (ctypes.c_void_p * max(len(boxed_args), 1))(
        *[ctypes.addressof(boxed_arg) for boxed_arg in boxed_args]
    )
    ensure_dynamic_shared_memory(function, shared_memory_bytes)
    result = _library().hipModuleLaunchKernel(
        HipFunction(function),
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_memory_bytes,
        ctypes.c_void_p(stream),
        arg_pointers,
        None,
    )
    check_hip(result, "hipModuleLaunchKernel")

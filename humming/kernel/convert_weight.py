import ctypes
import dataclasses
import math
from typing import ClassVar

import cuda.bindings.driver as cbd
import torch

from humming import dtypes
from humming.jit.runtime import KernelRuntime


@dataclasses.dataclass(kw_only=True)
class ConvertWeightKernel(KernelRuntime):
    name: ClassVar[str] = "convert_weight"
    source_dtype: dtypes.DataType
    target_dtype: dtypes.DataType
    check_only: bool = False

    def init_kernel(self):
        self.code = "#include <humming/kernel/convert_weight.cuh>\n"
        self.kernel_expr = (
            f"convert_weight_kernel<{self.source_dtype.to_cpp_str()}, "
            f"{self.target_dtype.to_cpp_str()}, {str(self.check_only).lower()}>"
        )
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
        self.prepare()

    def __call__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        if self.check_only:
            outputs.fill_(1)
        num_elements = inputs.nelement()
        if num_elements == 0:
            return

        func = self.load_cubin()
        config = cbd.CUlaunchConfig()
        words_per_thread = self.source_dtype.num_bits // math.gcd(self.source_dtype.num_bits, 32)
        num_threads = num_elements // words_per_thread
        config.gridDimX = (num_threads + 255) // 256
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 256
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(inputs.device).cuda_stream
        arg_values = (inputs.data_ptr(), outputs.data_ptr(), num_elements)
        cbd.cuLaunchKernelEx(config, func, (arg_values, self.arg_types), 0)

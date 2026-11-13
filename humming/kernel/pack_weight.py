import ctypes
import dataclasses
import math
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/pack_weight.cuh>

""")


@dataclasses.dataclass(kw_only=True)
class PackWeightKernel(KernelRuntime):
    name: ClassVar[str] = "pack_weight"
    num_bits: int
    source_bits: int = 32

    def init_kernel(self):
        self.code = CODE_TEMPLATE.render(num_bits=self.num_bits)
        assert self.source_bits in (8, 32)
        scalar_type = "uint8_t" if self.source_bits == 8 else "uint32_t"
        self.kernel_expr = f"pack_weight_kernel<{self.num_bits}, {scalar_type}>"
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
        self.prepare()

    def __call__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        num_elements = inputs.nelement()
        if num_elements == 0:
            return
        func = self.load_cubin()
        device = inputs.device
        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(num_elements / (32 * 32))
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (inputs.data_ptr(), outputs.data_ptr(), num_elements)

        cbd.cuLaunchKernelEx(config, func, (arg_values, self.arg_types), 0)

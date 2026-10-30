import ctypes
import dataclasses
import math
from typing import ClassVar

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

    def init_kernel(self):
        self.code = CODE_TEMPLATE.render(num_bits=self.num_bits)
        self.kernel_expr = f"pack_weight_kernel<{self.num_bits}>"
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
        self.prepare()

    def __call__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        func = self.load_cubin()
        device = inputs.device
        num_elements = inputs.nelement()
        grid = (math.ceil(num_elements / (32 * 32)), 1, 1)
        block = (32, 1, 1)
        stream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (inputs.data_ptr(), outputs.data_ptr(), num_elements)

        self.launch(func, grid, block, stream, arg_values)

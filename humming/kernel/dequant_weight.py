import ctypes
import dataclasses
import math
from typing import ClassVar

import torch

from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = """
#include <humming/kernel/dequant_weight.cuh>

"""


@dataclasses.dataclass(kw_only=True)
class DequantKernel(KernelRuntime):
    disable_fast_math: ClassVar[bool] = True
    name: ClassVar[str] = "dequant_unpacked_fp_type"

    def init_kernel(self):
        self.code = CODE_TEMPLATE
        self.kernel_expr = "dequant_unpacked_fp_type"
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_bool,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        exponent_bits: int,
        mantissa_bits: int,
        is_signed: bool,
    ):
        func = self.load_cubin()
        device = inputs.device
        total_size = inputs.nelement()
        grid = (math.ceil(total_size / (32 * 32)), 1, 1)
        block = (32, 1, 1)
        stream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            total_size,
            exponent_bits,
            mantissa_bits,
            is_signed,
        )

        self.launch(func, grid, block, stream, arg_values)
        return outputs

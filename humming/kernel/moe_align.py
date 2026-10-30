"""Single-workgroup ``moe_align_block_size`` kernel (``humming/kernel/moe_align.cuh``).

Drop-in for the torch implementation in ``humming.ops.moe`` (same buffers and contract) at a
fraction of its cost: one launch instead of ~25 small torch kernels (sort, scatter, cumsum,
searchsorted, ...), which matters at decode where the GEMMs themselves take a few microseconds.
"""

import ctypes
import dataclasses
from typing import ClassVar

import jinja2
import torch

from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/moe_align.cuh>
""")

NUM_THREADS = 1024
MAX_EXPERTS = NUM_THREADS


@dataclasses.dataclass(kw_only=True)
class MoeAlignKernel(KernelRuntime):
    name: ClassVar[str] = "moe_align_kernel"
    int64_ids: bool

    def init_kernel(self):
        self.code = CODE_TEMPLATE.render()
        self.kernel_expr = f"moe_align_kernel<{str(self.int64_ids).lower()}>"
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        )
        self.prepare()

    def __call__(
        self,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_padded: torch.Tensor,
        num_experts: int,
        block_size: int,
        ignore_invalid_experts: bool,
    ) -> None:
        assert topk_ids.is_contiguous() and num_experts <= MAX_EXPERTS
        assert expert_map is None or (expert_map.dtype == torch.int32 and expert_map.is_contiguous())
        func = self.load_cubin()
        stream = torch.cuda.current_stream(topk_ids.device).cuda_stream
        arg_values = (
            topk_ids.data_ptr(),
            0 if expert_map is None else expert_map.data_ptr(),
            sorted_ids.data_ptr(),
            expert_ids.data_ptr(),
            num_tokens_padded.data_ptr(),
            topk_ids.numel(),
            num_experts,
            block_size,
            sorted_ids.numel(),
            expert_ids.numel(),
            int(ignore_invalid_experts),
        )
        self.launch(func, (1, 1, 1), (NUM_THREADS, 1, 1), stream, arg_values)

"""ROCm gfx950 MFMA MoE GEMM (``humming/kernel/humming_mfma_moe.cuh``).

Same tile / dtype parameters as the dense ``HummingMfmaKernel``; ``gemm_type`` selects the MoE
row layout (indexed, grouped_contiguous, grouped_masked). The launcher recognises the kernel by
the ``MFMA_MOE_TYPE`` symbol (csrc/launcher/mfma_moe.h).
"""

import dataclasses
from typing import ClassVar

import jinja2

from humming import dtypes
from humming.config import GemmType
from humming.kernel.humming_mfma import HummingMfmaKernel
from humming.tune.gfx950 import MfmaTile

MOE_TYPE_IDS = {
    GemmType.INDEXED: 1,
    GemmType.GROUPED_CONTIGUOUS: 2,
    GemmType.GROUPED_MASKED: 3,
}

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/humming_mfma_moe.cuh>

using HummingMfmaTile = MfmaTileConfig<
    {{tile.warps_m}}, {{tile.warps_n}}, {{tile.tiles_m}}, {{tile.tiles_n}}, {{tile.stages}},
    {{tile.lds_a | lower}}, {{tile.scale_b | lower}}>;

// Read by the C++ launcher (csrc/launcher/mfma.h, mfma_moe.h).
extern "C" __constant__ uint32_t MFMA_BLOCK_M = HummingMfmaTile::kBlockM;
extern "C" __constant__ uint32_t MFMA_BLOCK_N = HummingMfmaTile::kBlockN;
extern "C" __constant__ uint32_t MFMA_NUM_THREADS = HummingMfmaTile::kNumThreads;
extern "C" __constant__ uint32_t MFMA_A_DTYPE_ID = {{a_dtype_id}};
extern "C" __constant__ uint32_t MFMA_NUM_BITS = {{num_bits}};
extern "C" __constant__ uint32_t MFMA_GROUP_SIZE = {{group_size}};
extern "C" __constant__ uint32_t MFMA_HAS_ZERO_POINT = {{has_zero_point | int}};
extern "C" __constant__ uint32_t MFMA_MAX_SPLITS = {{max_splits}};
extern "C" __constant__ uint32_t MFMA_MOE_TYPE = {{moe_type}};
""")


@dataclasses.dataclass(kw_only=True)
class HummingMfmaMoeKernel(HummingMfmaKernel):
    name: ClassVar[str] = "humming_mfma_moe"

    gemm_type: str = GemmType.INDEXED.value

    def init_kernel(self) -> None:
        a_dtype = dtypes.DataType.from_str(self.a_dtype)
        assert a_dtype in (dtypes.float16, dtypes.bfloat16)
        assert 1 <= self.num_bits <= 8 and self.group_size % 32 == 0
        moe_type = MOE_TYPE_IDS[GemmType(self.gemm_type)]
        tile = MfmaTile(
            self.warps_m, self.warps_n, self.tiles_m, self.tiles_n, self.stages, self.lds_a, self.scale_b
        )
        self.code = CODE_TEMPLATE.render(
            tile=tile,
            a_dtype_id=a_dtype.id(),
            num_bits=self.num_bits,
            group_size=self.group_size,
            has_zero_point=self.has_zero_point,
            max_splits=self.max_splits,
            moe_type=moe_type,
        )
        element_a = "Float16" if a_dtype == dtypes.float16 else "BFloat16"
        has_zero_point = "true" if self.has_zero_point else "false"
        self.kernel_expr = (
            f"humming_mfma_moe<{element_a}, UInt{self.num_bits}, HummingMfmaTile, {self.group_size}, "
            f"{has_zero_point}, {moe_type}>"
        )
        self.prepare()
        from humming import ops

        self.kernel_id, self.kernel_name = ops.register_kernel(self.kernel_filename)

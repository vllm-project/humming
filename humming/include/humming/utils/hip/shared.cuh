#pragma once
// ROCm replacement for utils/ptx/shared.cuh: ldmatrix/stmatrix are NVIDIA warp-collective
// fragment layouts with no CDNA counterpart. The MFMA backend owns its LDS access patterns.

#include <humming/utils/hip/unsupported.cuh>

HUMMING_ROCM_UNSUPPORTED_STUB(void, ld_shared)
HUMMING_ROCM_UNSUPPORTED_STUB(void, ld_shared_dequant)
HUMMING_ROCM_UNSUPPORTED_STUB(void, st_shared)
HUMMING_ROCM_UNSUPPORTED_STUB(void, aiu_ld_shared)

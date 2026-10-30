"""ROCm-only tests: skipped entirely on CUDA builds."""

import pytest
import torch

ROCM_SKIP_REASON = "ROCm-only test (torch is not a ROCm build or no GPU is visible)"


def pytest_collection_modifyitems(config, items):
    is_rocm_with_gpu = torch.version.hip is not None and torch.cuda.is_available()
    if is_rocm_with_gpu:
        return
    skip_marker = pytest.mark.skip(reason=ROCM_SKIP_REASON)
    for item in items:
        if "tests/rocm" in str(item.fspath):
            item.add_marker(skip_marker)

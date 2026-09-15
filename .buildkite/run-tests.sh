#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export HUMMING_TEST_TUNING_SOURCE="${HUMMING_TEST_TUNING_SOURCE:-heuristic}"
case "${HUMMING_TEST_TUNING_SOURCE}" in
  heuristic|sampled|batch_invariant) ;;
  *)
    echo "Invalid HUMMING_TEST_TUNING_SOURCE: ${HUMMING_TEST_TUNING_SOURCE}" >&2
    exit 1
    ;;
esac

echo "--- Install test dependencies"
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates git build-essential
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin sh
fi

uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python --torch-backend cu130 \
  'torch==2.13.0' pytest -e .

mkdir -p test-results
export HUMMING_TEST_NUMERICAL_ERROR_LOG="${PWD}/test-results/numerical-errors.jsonl"

echo "--- Check GPU and toolchain"
nvidia-smi | tee test-results/nvidia-smi.txt
nvcc --version | tee test-results/nvcc.txt
uv pip freeze --python .venv/bin/python > test-results/packages.txt
.venv/bin/python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
assert torch.__version__.split("+")[0] == "2.13.0", "Expected PyTorch 2.13.0"
assert torch.version.cuda and int(torch.version.cuda.split(".")[0]) >= 13, "Expected CUDA 13+"
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "Expected one allocated GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo "--- Run tests (${HUMMING_TEST_TUNING_SOURCE})"
.venv/bin/python -m pytest tests -v --junitxml=test-results/pytest.xml

if [[ "${HUMMING_RUN_BENCHMARK:-0}" == "1" ]]; then
  echo "--- Benchmark dense W4A16 GEMM"
  .venv/bin/python benchmarks/bench_humming.py \
    --shape_n 8192 --shape_k 8192 \
    --a_dtype float16 --b_dtype int4 --bs_dtype float16 --c_dtype float16 \
    --shape_m_list 1 16 64 256 1024 \
    --output_file test-results/benchmark-w4a16.json
fi

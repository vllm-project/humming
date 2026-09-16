#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "--- :nvidia: GPU Info"
nvidia-smi

echo "--- Install test dependencies"
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates git build-essential
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin sh
fi

uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python \
  pytest safetensors jinja2 cuda-bindings tqdm tabulate
uv pip install --python .venv/bin/python --no-deps -e .

echo "--- Run tests"
mkdir -p test-results
.venv/bin/python -m pytest tests -v --junitxml=test-results/pytest.xml

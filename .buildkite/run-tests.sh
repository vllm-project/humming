#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "--- CPU and compilation limits"
python3 humming/utils/cpu.py

echo "--- :nvidia: GPU Info"
nvidia-smi

echo "--- Install test dependencies"
for tool in curl git make g++; do
  command -v "${tool}" >/dev/null
done

export HOME="${TMPDIR:-/tmp}/humming-home"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"
mkdir -p "${HOME}" "${UV_CACHE_DIR}"
if ! command -v uv >/dev/null 2>&1; then
  uv_install_dir="${TMPDIR:-/tmp}/uv-bin"
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="${uv_install_dir}" UV_NO_MODIFY_PATH=1 sh
  export PATH="${uv_install_dir}:${PATH}"
fi

uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python \
  pytest safetensors jinja2 cuda-bindings tqdm tabulate
uv pip install --python .venv/bin/python --no-deps -e .

echo "--- Run tests"
if [[ "${HUMMING_TEST_TUNING_SOURCE:-}" == sampled* ]]; then
  export HUMMING_TEST_TUNING_SEED="${HUMMING_TEST_TUNING_SEED:-${BUILDKITE_BUILD_NUMBER:-0}}"
  echo "Sampled tuning seed: ${HUMMING_TEST_TUNING_SEED}"
fi
mkdir -p test-results
if [[ $# -eq 0 ]]; then
  set -- tests
fi
if [[ "${HUMMING_TEST_TUNING_SOURCE:-}" == sampled* && $# -eq 1 && -d "$1" ]]; then
  mapfile -d '' test_files < <(find "$1" -type f -name 'test_*.py' -print0 | sort -z)
  if [[ ${#test_files[@]} -eq 0 ]]; then
    echo "No test files found in $1" >&2
    exit 1
  fi
  test_status=0
  for test_file in "${test_files[@]}"; do
    report_name="${test_file//\//_}"
    echo "--- Run ${test_file}"
    .venv/bin/python -m pytest "${test_file}" -v \
      --junitxml="test-results/pytest-${report_name%.py}.xml" || test_status=1
  done
  exit "${test_status}"
fi
.venv/bin/python -m pytest "$@" -v --junitxml=test-results/pytest.xml

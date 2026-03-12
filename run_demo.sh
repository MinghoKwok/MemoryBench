#!/usr/bin/env bash
set -euo pipefail

# One-command runner for Comic Memory Benchmark demo.
# You can override defaults via env vars or CLI args.
#
# Examples:
#   bash run_demo.sh
#   bash run_demo.sh --single-image-test
#   MODEL_PATH=/path/to/local/model bash run_demo.sh --max-new-tokens 160

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/common/users/mg1998/models/Qwen2-VL-2B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-129600}" # 360x360

if [[ ! -f "${PROJECT_ROOT}/run_pipeline.py" ]]; then
  echo "[ERROR] run_pipeline.py not found in ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] Local model path not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH to your local Qwen2-VL directory." >&2
  exit 1
fi

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] max_new_tokens=${MAX_NEW_TOKENS}, max_image_pixels=${MAX_IMAGE_PIXELS}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/run_pipeline.py" \
  --model-path "${MODEL_PATH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-image-pixels "${MAX_IMAGE_PIXELS}" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"

TASK_CONFIG="${TASK_CONFIG:-${BENCHMARK_DIR}/config/tasks_external/home_renovation_interior_design.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${BENCHMARK_DIR}/config/models/gpt_4o_mini.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-${BENCHMARK_DIR}/config/methods/m2a.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DIR}/runs}"
MEMORYBENCH_CONDA_ENV="${MEMORYBENCH_CONDA_ENV:-memorybench}"
M2A_EMBEDDING_URL="${M2A_EMBEDDING_URL:-http://127.0.0.1:8050/v1}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"
MODE="${MODE:-}"
LOG_DIR="${LOG_DIR:-${BENCHMARK_DIR}/logs}"
TASK_BASENAME="$(basename "${TASK_CONFIG}" .yaml)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_official_m2a_${TASK_BASENAME}.log}"

if [[ -f "${REPO_ROOT}/.env.local" ]]; then
  set -a
  source "${REPO_ROOT}/.env.local"
  set +a
fi

# The local placeholder token is non-ASCII and breaks HF/httpx headers.
unset HUGGING_FACE_HUB_TOKEN

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Load ${REPO_ROOT}/.env.local first." >&2
  exit 1
fi

if ! MODEL_LIST="$(curl -fsS "${M2A_EMBEDDING_URL}/models")"; then
  echo "SigLIP2 vLLM is not reachable at ${M2A_EMBEDDING_URL}." >&2
  echo "Start it with: ${BENCHMARK_DIR}/scripts/start_siglip2_vllm.sh" >&2
  exit 1
fi

if [[ "${MODEL_LIST}" != *"siglip2-base-patch16-384"* ]]; then
  echo "SigLIP2 vLLM is reachable, but the served model name is unexpected." >&2
  echo "Expected to find siglip2-base-patch16-384 in ${M2A_EMBEDDING_URL}/models." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Running official agentic M2A"
echo "Task config: ${TASK_CONFIG}"
echo "Model config: ${MODEL_CONFIG} (used for run metadata/output naming)"
echo "Method config: ${METHOD_CONFIG} (pins the actual M2A LLM to gpt-4o-mini)"
echo "Embedding service: ${M2A_EMBEDDING_URL}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"

CMD=(
  conda run --no-capture-output -n "${MEMORYBENCH_CONDA_ENV}"
  python -u -m Benchmark_Pipeline.run_benchmark
  --task-config "${TASK_CONFIG}"
  --model-config "${MODEL_CONFIG}"
  --method-config "${METHOD_CONFIG}"
  --output-root "${OUTPUT_ROOT}"
)

if [[ "${MAX_QUESTIONS}" != "0" ]]; then
  CMD+=(--max-questions "${MAX_QUESTIONS}")
fi

if [[ -n "${MODE}" ]]; then
  CMD+=(--mode "${MODE}")
fi

cd "${REPO_ROOT}"
exec "${CMD[@]}"

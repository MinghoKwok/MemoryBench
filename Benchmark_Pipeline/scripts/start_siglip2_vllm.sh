#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-google/siglip2-base-patch16-384}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-siglip2-base-patch16-384}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8050}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-vllm}"

# The placeholder token in local shells can break HF/httpx header encoding.
unset HUGGING_FACE_HUB_TOKEN

echo "Starting SigLIP2 vLLM service on ${HOST}:${PORT}"
echo "Model repo: ${MODEL_REPO}"
echo "Served model name: ${SERVED_MODEL_NAME}"

exec conda run -n "${VLLM_CONDA_ENV}" \
  vllm serve "${MODEL_REPO}" \
  --task embed \
  --runner pooling \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-request-chat-template

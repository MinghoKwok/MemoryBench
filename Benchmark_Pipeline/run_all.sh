#!/bin/bash
# ============================================================
# MemEye Benchmark Runner — run a method × model across ALL datasets
# ============================================================
#
# Usage:
#   bash run_all.sh --method <method> --model <model> [options]
#
# Examples:
#   bash run_all.sh --method full_context_multimodal --model gemini_2_5_flash_lite
#   bash run_all.sh --method m2a_gemini --model gemini_2_5_flash_lite --datasets "card_playlog_test,brand_memory_test"
#   bash run_all.sh --method mma --model gpt_4_1_nano --max-questions 5
#   bash run_all.sh --list   # show available methods, models, datasets
#
# Environment:
#   Requires conda env 'memorybench'. API keys loaded from ../.env.local
#   For Gemini methods: set GEMINI_API_KEY (or use ../.env.local)
#   For OpenAI methods: set OPENAI_API_KEY (or use ../.env.local)
#
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ----
METHOD=""
MODEL=""
DATASETS=""
MAX_QUESTIONS=0
EXTRA_ARGS=""

# ---- All datasets ----
ALL_DATASETS=(
  animation_viewing_companion
  brand_memory_test
  card_playlog_test
  comic_reading_companion
  home_renovation_interior_design
  multi_scene_visual_case_archive_assistant
  outdoor_navigation_route_memory_assistant
  personal_health_dashboard_assistant
  social_chat_memory_test
)

# ---- Parse arguments ----
show_list() {
  echo ""
  echo "Available models:"
  for f in config/models/*.yaml; do echo "  $(basename "$f" .yaml)"; done
  echo ""
  echo "Available methods:"
  for f in config/methods/*.yaml; do echo "  $(basename "$f" .yaml)"; done
  echo ""
  echo "Available datasets:"
  for ds in "${ALL_DATASETS[@]}"; do echo "  $ds"; done
  echo ""
  exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --method)    METHOD="$2";        shift 2 ;;
    --model)     MODEL="$2";         shift 2 ;;
    --datasets)  DATASETS="$2";      shift 2 ;;
    --max-questions) MAX_QUESTIONS="$2"; shift 2 ;;
    --list)      show_list ;;
    -h|--help)
      echo "Usage: bash run_all.sh --method <method> --model <model> [--datasets ds1,ds2] [--max-questions N]"
      echo "       bash run_all.sh --list"
      exit 0 ;;
    *)
      EXTRA_ARGS="$EXTRA_ARGS $1"
      shift ;;
  esac
done

if [[ -z "$METHOD" || -z "$MODEL" ]]; then
  echo "Error: --method and --model are required."
  echo "Run 'bash run_all.sh --list' to see available options."
  exit 1
fi

# ---- Validate configs exist ----
METHOD_CFG="config/methods/${METHOD}.yaml"
MODEL_CFG="config/models/${MODEL}.yaml"

if [[ ! -f "$METHOD_CFG" ]]; then
  echo "Error: method config not found: $METHOD_CFG"
  echo "Run 'bash run_all.sh --list' to see available methods."
  exit 1
fi
if [[ ! -f "$MODEL_CFG" ]]; then
  echo "Error: model config not found: $MODEL_CFG"
  echo "Run 'bash run_all.sh --list' to see available models."
  exit 1
fi

# ---- Load API keys ----
if [[ -f "../.env.local" ]]; then
  source ../.env.local
fi
unset HUGGING_FACE_HUB_TOKEN 2>/dev/null

# ---- Sync latest data from HuggingFace ----
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Syncing latest data from HuggingFace..."
  conda run --no-capture-output -n memorybench \
    env HF_TOKEN="$HF_TOKEN" \
    python sync_hf_data.py pull 2>&1 | tail -3
  echo ""
fi

# For *_gemini methods, redirect OpenAI SDK to Gemini endpoint
if [[ "$METHOD" == *"gemini"* ]] || [[ "$MODEL" == *"gemini"* ]]; then
  export OPENAI_API_KEY="${GEMINI_API_KEY:-$OPENAI_API_KEY}"
  export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
fi

# For *_openrouter methods or openrouter models, redirect OpenAI SDK to OpenRouter
if [[ "$METHOD" == *"openrouter"* ]] || [[ "$MODEL" == *"openrouter"* ]]; then
  export OPENAI_API_KEY="${OPENROUTER_API_KEY:-$OPENAI_API_KEY}"
  export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
  export OPENAI_API_BASE="https://openrouter.ai/api/v1"
fi

# ---- Resolve dataset list ----
if [[ -n "$DATASETS" ]]; then
  IFS=',' read -ra DS_ARRAY <<< "$DATASETS"
else
  DS_ARRAY=("${ALL_DATASETS[@]}")
fi

# ---- Build extra flags ----
EXTRA_FLAGS=""
if [[ "$MAX_QUESTIONS" -gt 0 ]]; then
  EXTRA_FLAGS="--max-questions $MAX_QUESTIONS"
fi

# ---- Run ----
TOTAL=${#DS_ARRAY[@]}
SUCCESS=0
FAIL=0
SKIP=0

echo "============================================"
echo " MemEye Benchmark"
echo " Method:  $METHOD"
echo " Model:   $MODEL"
echo " Datasets: $TOTAL"
echo " Started:  $(date)"
echo "============================================"
echo ""

for i in "${!DS_ARRAY[@]}"; do
  ds="${DS_ARRAY[$i]}"
  idx=$((i + 1))
  TASK_CFG="config/tasks_external/${ds}.yaml"

  if [[ ! -f "$TASK_CFG" ]]; then
    echo "[$idx/$TOTAL] SKIP $ds — no task config"
    SKIP=$((SKIP + 1))
    continue
  fi

  echo "[$idx/$TOTAL] $ds × $METHOD"

  conda run --no-capture-output -n memorybench \
    env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
        OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
        OPENAI_API_BASE="${OPENAI_API_BASE:-}" \
        GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
        HF_TOKEN="${HF_TOKEN:-}" \
    python run_benchmark.py \
      --task-config "$TASK_CFG" \
      --model-config "$MODEL_CFG" \
      --method-config "$METHOD_CFG" \
      $EXTRA_FLAGS $EXTRA_ARGS \
    2>&1 | grep -E '^\[INFO\] (QA|Saved)|^\[MCQ\]|Error|error' | tail -3

  if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    SUCCESS=$((SUCCESS + 1))
    echo "  → done"
  else
    FAIL=$((FAIL + 1))
    echo "  → FAILED"
  fi
  echo ""
done

echo "============================================"
echo " Finished: $(date)"
echo " Success: $SUCCESS | Failed: $FAIL | Skipped: $SKIP | Total: $TOTAL"
echo "============================================"

#!/bin/bash
# Phase 1: Full benchmark sweep — 9 datasets × 6 methods × gpt-4.1-nano
# Usage: bash scripts/run_phase1.sh
# Runs each dataset sequentially through 6 methods.
# Launch multiple instances of this script (with different DATASETS) to parallelize.

set -euo pipefail

cd "$(dirname "$0")/.."
source ../.env.local
unset HUGGING_FACE_HUB_TOKEN

MODEL=config/models/gemini_2_5_flash_lite.yaml

METHODS=(
  full_context_multimodal
  full_context_text_only
  semantic_rag_multimodal
  semantic_rag_text_only
  m2a
  mma
)

DATASETS=(
  cartoon_entertainment_companion
  brand_memory_test
  card_playlog_test
  home_renovation_interior_design
  multi_scene_visual_case_archive_assistant
  outdoor_navigation_route_memory_assistant
  personal_health_dashboard_assistant
  social_chat_memory_test
)

# Allow selecting a subset via CLI args: bash run_phase1.sh 0 2  (runs datasets 0,1,2)
START=${1:-0}
END=${2:-8}

for i in $(seq "$START" "$END"); do
  ds="${DATASETS[$i]}"
  echo ""
  echo "=========================================="
  echo "Dataset: $ds"
  echo "=========================================="
  for method in "${METHODS[@]}"; do
    echo "--- $ds × $method ---"
    conda run --no-capture-output -n memorybench \
      env GEMINI_API_KEY="$GEMINI_API_KEY" \
          OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
          HF_TOKEN="${HF_TOKEN:-}" \
      python run_benchmark.py \
        --task-config "config/tasks_external/${ds}.yaml" \
        --model-config "$MODEL" \
        --method-config "config/methods/${method}.yaml" \
      2>&1 | tail -5
    echo "--- done ---"
  done
done

echo ""
echo "Phase 1 complete."

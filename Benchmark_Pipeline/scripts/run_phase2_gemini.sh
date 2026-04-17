#!/bin/bash
# Phase 2: Run remaining methods with Gemini via OpenAI-compatible endpoint
# All methods use OpenAI SDK internally — we redirect via env vars
set +e  # continue on errors

cd "$(dirname "$0")/.."
source ../.env.local
unset HUGGING_FACE_HUB_TOKEN

# Redirect OpenAI SDK to Gemini (keys loaded from ../.env.local above)
export OPENAI_API_KEY="${GEMINI_API_KEY}"
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

MODEL=config/models/gemini_2_5_flash_lite.yaml

METHODS=(a_mem_gemini reflexion_gemini simplemem_gemini simplemem_multimodal_gemini memoryos_gemini mirix_gemini m2a_gemini mma_gemini)

DATASETS=(
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

LOGFILE="logs/phase2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Phase 2 started at $(date)" | tee "$LOGFILE"
echo "Methods: ${METHODS[*]}" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

TOTAL=0
SUCCESS=0
FAIL=0

for method in "${METHODS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo "=== [$TOTAL] $ds × $method ===" | tee -a "$LOGFILE"
    
    conda run --no-capture-output -n memorybench \
      env OPENAI_API_KEY="$OPENAI_API_KEY" \
          OPENAI_BASE_URL="$OPENAI_BASE_URL" \
          GEMINI_API_KEY="$GEMINI_API_KEY" \
      python run_benchmark.py \
        --task-config "config/tasks_external/${ds}.yaml" \
        --model-config "$MODEL" \
        --method-config "config/methods/${method}.yaml" \
      >> "$LOGFILE" 2>&1
    
    if [ $? -eq 0 ]; then
      SUCCESS=$((SUCCESS + 1))
      echo "  ✓ SUCCESS" | tee -a "$LOGFILE"
    else
      FAIL=$((FAIL + 1))
      echo "  ✗ FAILED (see log)" | tee -a "$LOGFILE"
    fi
    echo "" | tee -a "$LOGFILE"
  done
done

echo "========================================" | tee -a "$LOGFILE"
echo "Phase 2 done at $(date)" | tee -a "$LOGFILE"
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAIL" | tee -a "$LOGFILE"

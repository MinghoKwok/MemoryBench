#!/bin/bash
# Usage: bash run_concat_quads.sh <method_config_name> [model_config]
# Runs 4-task concat groups for context-scale ablation.
# Group A: brand+social+cartoon+card
# Group B: health+home+multiscene+outdoor

set -u
cd "$(dirname "$0")"
source ../.env.local
unset HUGGING_FACE_HUB_TOKEN
export OPENAI_API_KEY
export GEMINI_API_KEY

METHOD_CFG="${1:?Usage: $0 <method_config> [model_config]}"
MODEL_CFG="${2:-gpt_5_4_mini}"

# Group A
QUAD_A_TASKS="brand_memory_test social_chat_memory_test cartoon_entertainment_companion card_playlog_test"
# Group B
QUAD_B_TASKS="personal_health_dashboard_assistant home_renovation_interior_design multi_scene_visual_case_archive_assistant outdoor_navigation_route_memory_assistant"

run_quad() {
  local group_name="$1"
  shift
  local tasks=("$@")

  for mode in mcq open; do
    local concat_name="concat_${group_name}"
    local suffix=""
    [ "$mode" = "open" ] && suffix="_open" && concat_name="concat_${group_name}_open"

    local concat_cfg="config/tasks_external/${concat_name}.yaml"
    local concat_dialog="data/dialog/${concat_name}.json"

    if [ ! -f "$concat_cfg" ]; then
      local task_cfgs=""
      for t in "${tasks[@]}"; do
        if [ "$mode" = "open" ]; then
          task_cfgs="$task_cfgs config/tasks_external/${t}_open.yaml"
        else
          task_cfgs="$task_cfgs config/tasks_external/${t}.yaml"
        fi
      done

      python3 create_concat_config.py \
        --task-configs $task_cfgs \
        --concat-name "$concat_name" \
        --eval-mode "$mode" \
        --output-cfg "$concat_cfg" \
        --output-dialog "$concat_dialog"
    fi

    local label="MCQ"
    [ "$mode" = "open" ] && label="Open"

    echo "=== ${label} ${METHOD_CFG} ${group_name} ==="
    conda run -n memorybench python run_benchmark.py \
      --task-config "$concat_cfg" \
      --model-config "config/models/${MODEL_CFG}.yaml" \
      --method-config "config/methods/${METHOD_CFG}.yaml" 2>&1 | grep -E "Saved|ERROR" | tail -3
    echo ""
  done
}

run_quad "quad_a" $QUAD_A_TASKS
run_quad "quad_b" $QUAD_B_TASKS

echo "=== COMPLETE ==="

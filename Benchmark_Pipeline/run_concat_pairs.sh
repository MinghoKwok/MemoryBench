#!/bin/bash
# Usage: bash run_concat_pairs.sh <method_config_name> [model_config]
# Runs 2-task concat pairs for context-scale ablation.
# Pairs: brand+social, cartoon+card, health+home, multiscene+outdoor

set -u
cd "$(dirname "$0")"
source ../.env.local
unset HUGGING_FACE_HUB_TOKEN
export OPENAI_API_KEY
export GEMINI_API_KEY

METHOD_CFG="${1:?Usage: $0 <method_config> [model_config]}"
MODEL_CFG="${2:-gpt_5_4_mini}"

run_pair() {
  local pair_name="$1"
  local task1="$2"
  local task2="$3"

  for mode in mcq open; do
    local concat_name="concat_${pair_name}"
    [ "$mode" = "open" ] && concat_name="concat_${pair_name}_open"

    local concat_cfg="config/tasks_external/${concat_name}.yaml"
    local concat_dialog="data/dialog/${concat_name}.json"

    if [ ! -f "$concat_cfg" ]; then
      local t1_cfg t2_cfg
      if [ "$mode" = "open" ]; then
        t1_cfg="config/tasks_external/${task1}_open.yaml"
        t2_cfg="config/tasks_external/${task2}_open.yaml"
      else
        t1_cfg="config/tasks_external/${task1}.yaml"
        t2_cfg="config/tasks_external/${task2}.yaml"
      fi

      python3 create_concat_config.py \
        --task-configs "$t1_cfg" "$t2_cfg" \
        --concat-name "$concat_name" \
        --eval-mode "$mode" \
        --output-cfg "$concat_cfg" \
        --output-dialog "$concat_dialog"
    fi

    local label="MCQ"
    [ "$mode" = "open" ] && label="Open"

    echo "=== ${label} ${METHOD_CFG} ${pair_name} ==="
    conda run -n memorybench python run_benchmark.py \
      --task-config "$concat_cfg" \
      --model-config "config/models/${MODEL_CFG}.yaml" \
      --method-config "config/methods/${METHOD_CFG}.yaml" 2>&1 | grep -E "Saved|ERROR" | tail -3
    echo ""
  done
}

run_pair "brand_social" "brand_memory_test" "social_chat_memory_test"
run_pair "cartoon_card" "cartoon_entertainment_companion" "card_playlog_test"
run_pair "health_home" "personal_health_dashboard_assistant" "home_renovation_interior_design"
run_pair "multiscene_outdoor" "multi_scene_visual_case_archive_assistant" "outdoor_navigation_route_memory_assistant"

echo "=== COMPLETE ==="

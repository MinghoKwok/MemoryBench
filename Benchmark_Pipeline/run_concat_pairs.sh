#!/bin/bash
# Usage: bash run_concat_pairs.sh <method_config_name> [model_config]
# Example: bash run_concat_pairs.sh full_context_multimodal gpt_5_4_mini
#
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

# Pair definitions: name, task1, task2
PAIRS=(
  "brand_social|brand_memory_test|social_chat_memory_test"
  "cartoon_card|cartoon_entertainment_companion|card_playlog_test"
  "health_home|personal_health_dashboard_assistant|home_renovation_interior_design"
  "multiscene_outdoor|multi_scene_visual_case_archive_assistant|outdoor_navigation_route_memory_assistant"
)

ERRORS=""

for pair_def in "${PAIRS[@]}"; do
  IFS='|' read -r pair_name task1 task2 <<< "$pair_def"

  for mode in "" "_open"; do
    if [ -z "$mode" ]; then
      label="MCQ"
      t1_cfg="config/tasks_external/${task1}.yaml"
      t2_cfg="config/tasks_external/${task2}.yaml"
    else
      label="Open"
      t1_cfg="config/tasks_external/${task1}_open.yaml"
      t2_cfg="config/tasks_external/${task2}_open.yaml"
    fi

    # Check if concat task config exists, create if not
    concat_task="concat_${pair_name}${mode}"
    concat_cfg="config/tasks_external/${concat_task}.yaml"

    if [ ! -f "$concat_cfg" ]; then
      echo "[INFO] Creating concat config: $concat_cfg"
      python3 << PYEOF
import json, yaml, os

# Load both task configs
def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)

t1 = load_yaml("$t1_cfg")
t2 = load_yaml("$t2_cfg")

# Load both dialog JSONs and merge
d1_path = t1['dataset']['dialog_json']
d2_path = t2['dataset']['dialog_json']

# Resolve relative paths
if not os.path.isabs(d1_path):
    d1_path = os.path.join(os.getcwd(), d1_path)
if not os.path.isabs(d2_path):
    d2_path = os.path.join(os.getcwd(), d2_path)

d1 = json.load(open(d1_path))
d2 = json.load(open(d2_path))

merged = {
    "dataset_name": f"Concat: {d1.get('task_name','')} + {d2.get('task_name','')}",
    "task_name": "${concat_task}",
    "character_profile": d1.get("character_profile", {}),
    "multi_session_dialogues": d1["multi_session_dialogues"] + d2["multi_session_dialogues"],
    "human-annotated QAs": d1.get("human-annotated QAs", []) + d2.get("human-annotated QAs", []),
}

# Write merged dialog
merged_path = f"data/dialog/${concat_task}.json"
os.makedirs(os.path.dirname(merged_path), exist_ok=True)
with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

# Write task config
cfg = {
    "name": "${concat_task}",
    "dataset": {
        "dialog_json": merged_path,
        "image_root": t1["dataset"].get("image_root", "data/image"),
    },
    "eval": {
        "mode": "$( [ -z '$mode' ] && echo 'mcq' || echo 'open' )",
        "output_json": "output/results_${concat_task}.json",
        "max_questions": 0,
    },
}
with open("$concat_cfg", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Created: $concat_cfg ({len(merged['multi_session_dialogues'])} sessions, {len(merged['human-annotated QAs'])} QAs)")
PYEOF
    fi

    echo "=== ${label} ${METHOD_CFG} ${pair_name} ==="
    conda run -n memorybench python run_benchmark.py \
      --task-config "$concat_cfg" \
      --model-config "config/models/${MODEL_CFG}.yaml" \
      --method-config "config/methods/${METHOD_CFG}.yaml" 2>&1 | grep -E "Saved|ERROR" | tail -3

    if [ $? -ne 0 ]; then
      ERRORS="${ERRORS}\n${label} ${METHOD_CFG} ${pair_name}"
    fi
    echo ""
  done
done

echo "=== COMPLETE ==="
if [ -n "$ERRORS" ]; then
  echo "FAILED:"
  echo -e "$ERRORS"
fi

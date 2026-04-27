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

GROUPS=(
  "quad_a|brand_memory_test|social_chat_memory_test|cartoon_entertainment_companion|card_playlog_test"
  "quad_b|personal_health_dashboard_assistant|home_renovation_interior_design|multi_scene_visual_case_archive_assistant|outdoor_navigation_route_memory_assistant"
)

for group_def in "${GROUPS[@]}"; do
  IFS='|' read -r group_name task1 task2 task3 task4 <<< "$group_def"

  for mode in "" "_open"; do
    if [ -z "$mode" ]; then
      label="MCQ"
      eval_mode="mcq"
    else
      label="Open"
      eval_mode="open"
    fi

    concat_task="concat_${group_name}${mode}"
    concat_cfg="config/tasks_external/${concat_task}.yaml"

    if [ ! -f "$concat_cfg" ]; then
      echo "[INFO] Creating 4-task concat config: $concat_cfg"
      python3 << PYEOF
import json, yaml, os

def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)

def load_dialog(task_cfg):
    t = load_yaml(task_cfg)
    p = t['dataset']['dialog_json']
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    return json.load(open(p)), t['dataset'].get('image_root', 'data/image')

suffix = "$mode"
tasks = ["$task1", "$task2", "$task3", "$task4"]
all_sessions = []
all_qas = []
image_root = None

for task in tasks:
    if suffix:
        cfg_path = f"config/tasks_external/{task}_open.yaml"
    else:
        cfg_path = f"config/tasks_external/{task}.yaml"
    d, ir = load_dialog(cfg_path)
    all_sessions.extend(d["multi_session_dialogues"])
    all_qas.extend(d.get("human-annotated QAs", []))
    if image_root is None:
        image_root = ir

merged = {
    "dataset_name": "Concat 4-task: $group_name",
    "task_name": "$concat_task",
    "character_profile": {},
    "multi_session_dialogues": all_sessions,
    "human-annotated QAs": all_qas,
}

merged_path = f"data/dialog/${concat_task}.json"
os.makedirs(os.path.dirname(merged_path), exist_ok=True)
with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

cfg = {
    "name": "$concat_task",
    "dataset": {"dialog_json": merged_path, "image_root": image_root},
    "eval": {"mode": "$eval_mode", "output_json": "output/results_${concat_task}.json", "max_questions": 0},
}
with open("$concat_cfg", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Created: $concat_cfg ({len(all_sessions)} sessions, {len(all_qas)} QAs)")
PYEOF
    fi

    echo "=== ${label} ${METHOD_CFG} ${group_name} ==="
    conda run -n memorybench python run_benchmark.py \
      --task-config "$concat_cfg" \
      --model-config "config/models/${MODEL_CFG}.yaml" \
      --method-config "config/methods/${METHOD_CFG}.yaml" 2>&1 | grep -E "Saved|ERROR" | tail -3
    echo ""
  done
done

echo "=== COMPLETE ==="

#!/usr/bin/env python3
"""Create a concat task config by merging multiple task dialog JSONs."""
import argparse
import json
import os
import yaml


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-configs", nargs="+", required=True)
    parser.add_argument("--concat-name", required=True)
    parser.add_argument("--eval-mode", default="mcq")
    parser.add_argument("--output-cfg", required=True)
    parser.add_argument("--output-dialog", required=True)
    args = parser.parse_args()

    all_sessions = []
    all_qas = []
    image_root = None

    for tc in args.task_configs:
        t = load_yaml(tc)
        p = t["dataset"]["dialog_json"]
        if not os.path.isabs(p):
            p = os.path.normpath(p)
        d = json.load(open(p))
        all_sessions.extend(d["multi_session_dialogues"])
        all_qas.extend(d.get("human-annotated QAs", []))
        if image_root is None:
            image_root = t["dataset"].get("image_root", "data/image")

    merged = {
        "dataset_name": f"Concat: {args.concat_name}",
        "task_name": args.concat_name,
        "character_profile": {},
        "multi_session_dialogues": all_sessions,
        "human-annotated QAs": all_qas,
    }

    os.makedirs(os.path.dirname(args.output_dialog), exist_ok=True)
    with open(args.output_dialog, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    cfg = {
        "name": args.concat_name,
        "dataset": {"dialog_json": args.output_dialog, "image_root": image_root},
        "eval": {
            "mode": args.eval_mode,
            "output_json": f"output/results_{args.concat_name}.json",
            "max_questions": 0,
        },
    }
    os.makedirs(os.path.dirname(args.output_cfg), exist_ok=True)
    with open(args.output_cfg, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"Created: {args.output_cfg} ({len(all_sessions)} sessions, {len(all_qas)} QAs)")


if __name__ == "__main__":
    main()

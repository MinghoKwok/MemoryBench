#!/usr/bin/env python3
"""Fix Multi-Scene_Visual_Case_Archive_Assistant.json field names:
  - QA: 'points' → 'point'
  - Dialogue: 'images' → 'input_image' (ensure value is list)
"""
import json
import sys
from pathlib import Path

def fix(data):
    # Fix dialogues: images → input_image
    for session in data.get("multi_session_dialogues", []):
        for dlg in session.get("dialogues", []):
            if "images" in dlg:
                val = dlg.pop("images")
                if isinstance(val, str):
                    val = [val]
                dlg["input_image"] = val

    # Fix QAs: points → point
    for qa in data.get("human-annotated QAs", []):
        if "points" in qa and "point" not in qa:
            qa["point"] = qa.pop("points")

    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_multiscene_format.py <path_to_json> [--write]")
        sys.exit(1)

    path = Path(sys.argv[1])
    write = "--write" in sys.argv

    data = json.loads(path.read_text(encoding="utf-8"))
    fixed = fix(data)

    if write:
        path.write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Written: {path}")
    else:
        # Dry run: show what changed
        sessions = fixed["multi_session_dialogues"]
        img_count = sum(1 for s in sessions for d in s["dialogues"] if "input_image" in d)
        qas = fixed.get("human-annotated QAs", [])
        point_count = sum(1 for q in qas if "point" in q)
        has_old_images = any("images" in d for s in sessions for d in s["dialogues"])
        has_old_points = any("points" in q for q in qas)
        print(f"Dialogues with input_image: {img_count}")
        print(f"QAs with point: {point_count}/{len(qas)}")
        print(f"Residual 'images' fields: {has_old_images}")
        print(f"Residual 'points' fields: {has_old_points}")
        print("(pass --write to apply)")

if __name__ == "__main__":
    main()

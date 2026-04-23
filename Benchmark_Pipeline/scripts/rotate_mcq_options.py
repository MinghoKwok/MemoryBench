#!/usr/bin/env python3
"""Convert MCQ dialog JSONs to rotation format.

For each QA with N options, generates N rotations via circular shift so
that the correct answer lands at each position (A, B, C, …) exactly once.

Before:
    "options": {"A": "wrong1", "B": "wrong2", "C": "wrong3", "D": "correct"},
    "answer": "D"

After:
    "options": [
        {"A": "correct", "B": "wrong1", "C": "wrong2", "D": "wrong3", "answer": "A"},
        {"A": "wrong3", "B": "correct", "C": "wrong2", "D": "wrong1", "answer": "B"},  (circular shift)
        ...
    ],
    "answer": "D"   ← kept for backward reference / human readability

Usage:
    python scripts/rotate_mcq_options.py [--dry-run]
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path


CACHE_DIR = Path.home() / ".cache" / "memeye_hf" / "MemEye" / "data" / "dialog"

MCQ_FILES = [
    "All_Concat.json",
    "Brand_Memory_Test.json",
    "Card_Playlog_Test.json",
    "Cartoon_Entertainment_Companion.json",
    "Home_Renovation_Interior_Design.json",
    "Multi-Scene_Visual_Case_Archive_Assistant.json",
    "Outdoor_Navigation_Route_Memory_Assistant.json",
    "Personal_Health_Dashboard_Assistant.json",
    "Social_Chat_Memory_Test.json",
]


def rotate_options(options: dict, answer: str) -> list:
    """Generate circular-shift rotations of MCQ options.

    Given options {"A": v_a, "B": v_b, "C": v_c, "D": v_d} and answer="D",
    produces len(options) rotations where the correct answer occupies each
    position exactly once.

    Each rotation is a dict with keys A, B, C, D, answer.
    """
    keys = sorted(options.keys())  # ["A", "B", "C", "D"]
    n = len(keys)
    values = [options[k] for k in keys]

    # Find index of the correct answer
    answer_idx = keys.index(answer.strip().upper())
    correct_value = values[answer_idx]

    # Build list of distractor values (preserving order, excluding correct)
    distractors = [v for i, v in enumerate(values) if i != answer_idx]

    rotations = []
    for target_pos in range(n):
        # Place correct answer at target_pos, fill others with distractors
        # Use circular shift of distractors to vary their positions
        shifted_distractors = distractors[target_pos:] + distractors[:target_pos]
        new_values = []
        d_idx = 0
        for pos in range(n):
            if pos == target_pos:
                new_values.append(correct_value)
            else:
                new_values.append(shifted_distractors[d_idx])
                d_idx += 1
        rotation = OrderedDict()
        for k, v in zip(keys, new_values):
            rotation[k] = v
        rotation["answer"] = keys[target_pos]
        rotations.append(dict(rotation))

    return rotations


def process_file(filepath: Path, dry_run: bool = False) -> dict:
    """Process a single dialog JSON file, converting QA options to rotations."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    # Find QAs
    qa_key = None
    for candidate in ("human-annotated QAs", "human_annotated_qas", "qas"):
        if candidate in data and isinstance(data[candidate], list):
            qa_key = candidate
            break

    if qa_key is None:
        return {"file": filepath.name, "status": "no_qas_found", "converted": 0, "skipped": 0}

    converted = 0
    skipped = 0
    already_rotated = 0

    for qa in data[qa_key]:
        options = qa.get("options")
        answer = qa.get("answer", "")

        # Skip if already in rotation format
        if isinstance(options, list):
            already_rotated += 1
            continue

        if not isinstance(options, dict) or not answer:
            skipped += 1
            continue

        if answer.strip().upper() not in options:
            print(f"  WARN: answer '{answer}' not in options keys {list(options.keys())} "
                  f"for question_id={qa.get('question_id', '?')}", file=sys.stderr)
            skipped += 1
            continue

        rotations = rotate_options(options, answer)
        qa["options"] = rotations
        # Keep original answer for reference
        converted += 1

    if not dry_run and converted > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return {
        "file": filepath.name,
        "status": "ok",
        "converted": converted,
        "skipped": skipped,
        "already_rotated": already_rotated,
    }


def main():
    parser = argparse.ArgumentParser(description="Rotate MCQ options for position-bias debiasing")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--dir", type=str, default=str(CACHE_DIR), help="Directory containing dialog JSONs")
    args = parser.parse_args()

    dialog_dir = Path(args.dir)
    if not dialog_dir.exists():
        print(f"ERROR: Directory not found: {dialog_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Processing MCQ files in {dialog_dir}")
    print()

    total_converted = 0
    for filename in MCQ_FILES:
        filepath = dialog_dir / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} not found")
            continue
        result = process_file(filepath, dry_run=args.dry_run)
        total_converted += result["converted"]
        print(f"  {result['file']}: {result['converted']} converted, "
              f"{result['skipped']} skipped, {result['already_rotated']} already rotated")

    print(f"\nTotal: {total_converted} QAs converted to rotation format")
    if args.dry_run:
        print("(dry run — no files modified)")


if __name__ == "__main__":
    main()

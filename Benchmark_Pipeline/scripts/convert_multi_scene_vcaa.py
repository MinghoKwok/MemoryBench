"""
Convert Multi-Scene_Visual_Case_Archive_Assistant.json
from the upstream HF format to MemEye benchmark format.

Upstream structure:
  - sessions[].turns[]  (speaker-level, user/assistant alternating)
  - evaluation_queries[] (with evidence as image_ids or turn_ids)

Benchmark structure:
  - multi_session_dialogues[].dialogues[] (round-level, user+assistant paired)
  - human-annotated QAs[] (with clue as round_ids, point as binocular coords)

Rules applied:
  1. Consecutive user+assistant turn pairs become one "round".
  2. Round IDs use the format "S1:1", "S1:2", etc.
  3. Images referenced by image_id are mapped to relative file paths.
  4. Evidence (image_ids / turn_ids) are mapped to the round that contains them.
  5. "skills" are split into X-axis labels and descriptive tags; Y-axis is inferred
     from whether evidence spans multiple sessions.
  6. For open-ended QAs, answer comes from hidden_reference.gold_answer.
     For MCQs, answer comes from top-level "answer" or hidden_reference.answer.
  7. Q27 is dropped — it is a meta-question about dataset design, not a memory QA.
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


IMAGE_DIR_NAME = "Multi-Scene_Visual_Case_Archive_Assistant"


def pair_turns_into_rounds(
    session: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """
    Pair alternating user/assistant turns into rounds.

    Returns:
        dialogues: list of round dicts for multi_session_dialogues
        turn_to_round: mapping from turn_id -> round_id
        image_to_round: mapping from image_id -> round_id
    """
    turns = session["turns"]
    session_id = session["session_id"]
    dialogues: List[Dict[str, Any]] = []
    turn_to_round: Dict[str, str] = {}
    image_to_round: Dict[str, str] = {}

    round_idx = 0
    i = 0
    while i < len(turns):
        t = turns[i]
        if t["speaker"] != "user":
            # Defensive: skip orphan assistant turns (shouldn't happen per data)
            i += 1
            continue

        user_turn = t
        assistant_turn = None
        if i + 1 < len(turns) and turns[i + 1]["speaker"] == "assistant":
            assistant_turn = turns[i + 1]

        round_idx += 1
        round_id = f"{session_id}:{round_idx}"

        # Map turn_ids to this round
        turn_to_round[user_turn["turn_id"]] = round_id
        if assistant_turn:
            turn_to_round[assistant_turn["turn_id"]] = round_id

        # Collect images from user turn
        input_images: List[str] = []
        for img in user_turn.get("images") or []:
            image_id = img["image_id"]
            image_to_round[image_id] = round_id
            # Use relative path matching existing benchmark convention
            input_images.append(f"../image/{IMAGE_DIR_NAME}/{image_id}.png")

        dialogue_entry: Dict[str, Any] = OrderedDict()
        dialogue_entry["round"] = round_id
        dialogue_entry["user"] = user_turn["text"]
        dialogue_entry["assistant"] = assistant_turn["text"] if assistant_turn else ""
        if input_images:
            dialogue_entry["input_image"] = input_images

        dialogues.append(dialogue_entry)
        i += 2 if assistant_turn else 1

    return dialogues, turn_to_round, image_to_round


def infer_y_axis(session_ids: List[str], evidence_count: int) -> List[str]:
    """Infer Y-axis label from evidence span."""
    if len(session_ids) >= 3:
        return ["Y3"]
    if len(session_ids) == 2:
        return ["Y2"]
    # Single session
    if evidence_count >= 2:
        return ["Y2"]
    return ["Y1"]


def extract_x_labels(skills: List[str]) -> List[str]:
    """Extract X-axis labels from skills list."""
    x_labels = sorted(set(s for s in skills if s.startswith("X")))
    return x_labels if x_labels else ["X0"]


def resolve_answer(query: Dict[str, Any]) -> str:
    """
    Determine the gold answer for a query.
    - Open-ended: hidden_reference.gold_answer
    - MCQ: top-level "answer" (letter), fallback to hidden_reference.answer
    """
    hr = query.get("hidden_reference") or {}
    has_options = bool(query.get("options"))

    if has_options:
        # MCQ: answer is a letter (A/B/C/D)
        ans = query.get("answer", "")
        if not ans:
            ans = hr.get("answer", "")
        return ans
    else:
        # Open-ended: full text answer
        return hr.get("gold_answer", "") or hr.get("answer", "") or query.get("answer", "")


def convert(src_path: str, dst_path: str) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── Build global mappings ──
    all_turn_to_round: Dict[str, str] = {}
    all_image_to_round: Dict[str, str] = {}
    converted_sessions: List[Dict[str, Any]] = []

    for session in data["sessions"]:
        dialogues, turn_to_round, image_to_round = pair_turns_into_rounds(session)
        all_turn_to_round.update(turn_to_round)
        all_image_to_round.update(image_to_round)

        converted_session: Dict[str, Any] = OrderedDict()
        converted_session["session_id"] = session["session_id"]
        converted_session["dialogues"] = dialogues
        converted_sessions.append(converted_session)

    # ── Convert evaluation queries to QAs ──
    converted_qas: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for query in data["evaluation_queries"]:
        qid = query.get("query_id") or query.get("question_id", "")
        skills = query.get("skills") or []

        # Skip meta-questions that are not actual memory QAs
        if "dataset design understanding" in skills:
            skipped.append(qid)
            continue

        answer = resolve_answer(query)
        if not answer:
            skipped.append(qid)
            print(f"[WARN] {qid}: no answer found, skipping")
            continue

        # Map evidence → round_ids
        evidence = query.get("evidence") or []
        clue_rounds: List[str] = []
        for ev in evidence:
            if ev in all_image_to_round:
                clue_rounds.append(all_image_to_round[ev])
            elif ev in all_turn_to_round:
                clue_rounds.append(all_turn_to_round[ev])
            else:
                # Unknown evidence reference (e.g. "global design") — skip silently
                pass
        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_clues: List[str] = []
        for r in clue_rounds:
            if r not in seen:
                seen.add(r)
                unique_clues.append(r)
        clue_rounds = unique_clues

        # Derive session_ids from clue rounds
        session_ids = sorted(set(r.split(":")[0] for r in clue_rounds))

        # Build point (binocular MemEye coordinates)
        x_labels = extract_x_labels(skills)
        y_labels = infer_y_axis(session_ids, len(clue_rounds))

        qa_entry: Dict[str, Any] = OrderedDict()
        qa_entry["point"] = [x_labels, y_labels]
        qa_entry["question"] = query["question"]
        qa_entry["answer"] = answer
        qa_entry["session_id"] = session_ids
        qa_entry["clue"] = clue_rounds

        # Preserve MCQ options if present
        if query.get("options"):
            qa_entry["options"] = query["options"]

        converted_qas.append(qa_entry)

    # ── Assemble output ──
    output: Dict[str, Any] = OrderedDict()
    output["character_profile"] = {
        "description": data.get("global_case_context", ""),
        "source_dataset": data.get("dataset_name", ""),
        "task_name": data.get("task_name", ""),
    }
    output["multi_session_dialogues"] = converted_sessions
    output["human-annotated QAs"] = converted_qas

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted_sessions)} sessions, "
          f"{len(converted_qas)} QAs (skipped {len(skipped)}: {skipped})")
    print(f"Written to {dst_path}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "Benchmark_Pipeline/data/dialog/Multi-Scene_Visual_Case_Archive_Assistant.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else \
        "Benchmark_Pipeline/data/dialog/Multi-Scene_Visual_Case_Archive_Assistant_converted.json"
    convert(src, dst)

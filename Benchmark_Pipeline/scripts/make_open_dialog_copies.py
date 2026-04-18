import argparse
import copy
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "dialog"
TASK_CONFIG_DIR = REPO_ROOT / "config" / "tasks"
DEFAULT_MODEL = "gpt-4.1-mini"
BLOCKED_PATTERNS = (
    "pick the option",
    "pick an option",
    "which of these",
    "which statement",
    "answer with only the option letter",
    "option letter",
)

SYSTEM_PROMPT = """You rewrite multiple-choice benchmark questions into open-ended questions.

Rules:
- Preserve the exact meaning, entities, temporal relations, ambiguity cues, and scope.
- Remove all multiple-choice framing.
- Do not mention options, letters, or answer instructions.
- Do not reveal the answer.
- Produce a concise, natural open-ended question that can be answered directly by the provided correct answer text.
- Return JSON only with this shape: {"question": "..."}.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate open-ended dialog dataset copies and parallel local task configs."
    )
    parser.add_argument(
        "--dialog-dir",
        default=str(DATA_DIR),
        help="Directory containing canonical dialog JSON files.",
    )
    parser.add_argument(
        "--task-config-dir",
        default=str(TASK_CONFIG_DIR),
        help="Directory to write local open task configs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model used for rewriting question text (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--review-csv",
        default="/tmp/memeye_open_question_rewrites.csv",
        help="Path for the rewrite review CSV.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional list of input dataset filenames to process.",
    )
    return parser.parse_args()


def resolve_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def slugify_task_name(filename: str) -> str:
    stem = Path(filename).stem
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return normalized


def get_qas(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("human-annotated QAs", "human_annotated_qas", "qas"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("Dialog payload is missing a QA list under the canonical keys.")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def build_user_prompt(question: str, options: Dict[str, Any], answer_key: str, answer_text: str) -> str:
    rendered_options = "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
    return (
        f"Original question:\n{question}\n\n"
        f"Options:\n{rendered_options}\n\n"
        f"Correct option key: {answer_key}\n"
        f"Correct option text: {answer_text}\n\n"
        "Rewrite this into a single open-ended question that is directly answerable by the correct option text.\n"
        'Return JSON only: {"question": "..."}'
    )


def rewrite_question(
    client: OpenAI,
    model: str,
    question: str,
    options: Dict[str, Any],
    answer_key: str,
    answer_text: str,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(question, options, answer_key, answer_text),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Rewrite model returned empty content.")
    payload = json.loads(content)
    rewritten = str(payload.get("question", "")).strip()
    if not rewritten:
        raise ValueError("Rewrite model returned an empty question.")
    return rewritten


def validate_rewritten_question(
    original_question: str,
    rewritten_question: str,
    answer_text: str,
) -> None:
    normalized_question = _normalize(rewritten_question)
    if not normalized_question:
        raise ValueError("Rewritten question is empty.")
    for blocked in BLOCKED_PATTERNS:
        if blocked in normalized_question:
            raise ValueError(
                f"Rewritten question still contains blocked MCQ phrasing '{blocked}': {rewritten_question}"
            )
    if "answer with only the option letter" in normalized_question:
        raise ValueError(f"Rewritten question still includes option-letter instructions: {rewritten_question}")
    if not str(answer_text).strip():
        raise ValueError(f"Answer text is empty for question: {original_question}")


def convert_qa(
    client: OpenAI,
    model: str,
    dataset_name: str,
    qa: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    options = qa.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError(f"{dataset_name} {qa.get('question_id')}: expected non-empty options dict.")

    answer_key = str(qa.get("answer", "")).strip()
    if answer_key not in options:
        raise ValueError(
            f"{dataset_name} {qa.get('question_id')}: answer key '{answer_key}' missing from options."
        )

    original_question = str(qa.get("question", "")).strip()
    if not original_question:
        raise ValueError(f"{dataset_name} {qa.get('question_id')}: empty question.")

    answer_text = str(options[answer_key]).strip()
    rewritten_question = rewrite_question(
        client=client,
        model=model,
        question=original_question,
        options=options,
        answer_key=answer_key,
        answer_text=answer_text,
    )
    validate_rewritten_question(original_question, rewritten_question, answer_text)

    converted = copy.deepcopy(qa)
    converted["question"] = rewritten_question
    converted["answer"] = answer_text
    converted.pop("options", None)

    review_row = {
        "dataset": dataset_name,
        "question_id": str(qa.get("question_id", "")),
        "original_question": original_question,
        "rewritten_question": rewritten_question,
        "answer_key": answer_key,
        "answer_text": answer_text,
    }
    return converted, review_row


def convert_dataset(
    path: Path,
    client: OpenAI,
    model: str,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    payload = load_json(path)
    qas = get_qas(payload)
    converted_qas: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, str]] = []
    for qa in qas:
        converted, review_row = convert_qa(client, model, path.name, qa)
        converted_qas.append(converted)
        review_rows.append(review_row)

    updated = copy.deepcopy(payload)
    if "human-annotated QAs" in updated:
        updated["human-annotated QAs"] = converted_qas
    elif "human_annotated_qas" in updated:
        updated["human_annotated_qas"] = converted_qas
    else:
        updated["qas"] = converted_qas
    return updated, review_rows


def write_task_config(task_config_dir: Path, input_filename: str) -> Path:
    task_slug = slugify_task_name(input_filename)
    config_name = f"{task_slug}_open.yaml"
    config_path = task_config_dir / config_name
    dialog_name = f"{Path(input_filename).stem}_Open.json"
    output_name = f"results_{task_slug}_open.json"
    content = (
        f"name: {task_slug}_open\n\n"
        "dataset:\n"
        f"  dialog_json: data/dialog/{dialog_name}\n"
        "  image_root: data/image\n\n"
        "eval:\n"
        "  mode: open\n"
        f"  output_json: output/{output_name}\n"
        "  max_questions: 0\n"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")
    return config_path


def validate_open_dataset(source: Dict[str, Any], converted: Dict[str, Any], dataset_name: str) -> None:
    source_qas = get_qas(source)
    converted_qas = get_qas(converted)
    if len(source_qas) != len(converted_qas):
        raise ValueError(f"{dataset_name}: QA count mismatch after conversion.")

    for source_qa, converted_qa in zip(source_qas, converted_qas):
        options = source_qa.get("options")
        answer_key = str(source_qa.get("answer", "")).strip()
        if not isinstance(options, dict) or answer_key not in options:
            raise ValueError(f"{dataset_name}: source QA missing valid MCQ fields for validation.")
        expected_answer = str(options[answer_key]).strip()
        if "options" in converted_qa:
            raise ValueError(f"{dataset_name}: converted QA still has options.")
        if str(converted_qa.get("answer", "")).strip() != expected_answer:
            raise ValueError(f"{dataset_name}: converted QA answer does not match correct option text.")


def iter_input_files(dialog_dir: Path, only: Iterable[str]) -> List[Path]:
    only_set = {value for value in only if value}
    files = []
    for path in sorted(dialog_dir.glob("*.json")):
        if path.name.endswith("_Open.json"):
            continue
        if only_set and path.name not in only_set:
            continue
        files.append(path)
    return files


def write_review_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "question_id",
                "original_question",
                "rewritten_question",
                "answer_key",
                "answer_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dialog_dir = resolve_dir(args.dialog_dir)
    task_config_dir = resolve_dir(args.task_config_dir)
    review_csv = Path(args.review_csv).expanduser().resolve()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set before generating open-ended dataset copies.")
    client = OpenAI(api_key=api_key)

    review_rows: List[Dict[str, str]] = []
    input_files = iter_input_files(dialog_dir, args.only)
    if not input_files:
        raise ValueError("No input dialog JSON files selected for conversion.")

    for input_path in input_files:
        source_payload = load_json(input_path)
        converted_payload, dataset_rows = convert_dataset(input_path, client=client, model=args.model)
        validate_open_dataset(source_payload, converted_payload, input_path.name)

        output_path = input_path.with_name(f"{input_path.stem}_Open.json")
        dump_json(output_path, converted_payload)
        write_task_config(task_config_dir, input_path.name)
        review_rows.extend(dataset_rows)
        print(f"Wrote {output_path}")

    write_review_csv(review_csv, review_rows)
    print(f"Wrote rewrite review CSV to {review_csv}")


if __name__ == "__main__":
    main()

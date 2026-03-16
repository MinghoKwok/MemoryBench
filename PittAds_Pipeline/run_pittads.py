import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from router import QwenLocalRouter

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PittAds/Mem-Gallery style benchmark.")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--dialog-json", type=str, default="")
    p.add_argument("--image-root", type=str, default="")
    p.add_argument("--model-path", type=str, default="")
    p.add_argument("--max-new-tokens", type=int, default=0)
    p.add_argument("--output-json", type=str, default="")
    p.add_argument("--mode", type=str, choices=["open", "mcq", "both"], default="")
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SCRIPT_DIR / path).resolve()


def resolve_dataset_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    base_candidates = [base_dir, base_dir.parent, SCRIPT_DIR, Path.cwd()]
    candidates: List[Path] = []
    seen: set[str] = set()
    for base in base_candidates:
        candidate = (base / path).resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml(resolve_config_path(args.config))
    model = cfg.setdefault("model", {})
    dataset = cfg.setdefault("dataset", {})
    ev = cfg.setdefault("eval", {})

    if args.dialog_json:
        dataset["dialog_json"] = args.dialog_json
    if args.image_root:
        dataset["image_root"] = args.image_root
    if args.model_path:
        model["model_path"] = args.model_path
    if args.max_new_tokens:
        model["max_new_tokens"] = args.max_new_tokens
    if args.output_json:
        ev["output_json"] = args.output_json
    if args.mode:
        ev["mode"] = args.mode

    return cfg


def resolve_image_path(
    raw_image_path: str, dialog_json_path: Path, image_root: Optional[Path]
) -> str:
    cleaned = raw_image_path.replace("file://", "")
    source_path = Path(cleaned)
    candidates: List[Path] = []

    if source_path.is_absolute():
        candidates.append(source_path)
    else:
        candidates.append((dialog_json_path.parent / source_path).resolve())
        candidates.append((dialog_json_path.parent.parent / source_path).resolve())

        normalized = cleaned
        for prefix in ("../image/", "./image/", "image/", "data/image/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        if image_root is not None:
            candidates.append((image_root / normalized).resolve())
            candidates.append((image_root / source_path.name).resolve())

        default_image_root = (dialog_json_path.parent.parent / "image").resolve()
        candidates.append((default_image_root / normalized).resolve())

    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not resolve image path "
        f"'{raw_image_path}' from dialog file '{dialog_json_path}'. "
        f"Tried: {[str(p) for p in deduped]}"
    )


def build_rounds(
    dialog_data: Dict[str, Any], dialog_json_path: Path, image_root: Optional[Path]
) -> Dict[str, Dict[str, Any]]:
    rounds: Dict[str, Dict[str, Any]] = {}
    for sess in dialog_data.get("multi_session_dialogues", []):
        sid = sess.get("session_id", "")
        for d in sess.get("dialogues", []):
            rid = d.get("round", "")
            images: List[str] = []
            for rel in d.get("input_image", []) or []:
                images.append(resolve_image_path(rel, dialog_json_path, image_root))
            rounds[rid] = {
                "session_id": sid,
                "user": d.get("user", ""),
                "assistant": d.get("assistant", ""),
                "images": images,
            }
    return rounds


def session_order(dialog_data: Dict[str, Any]) -> List[str]:
    return [s.get("session_id", "") for s in dialog_data.get("multi_session_dialogues", [])]


def build_history_for_qa(
    dialog_data: Dict[str, Any], qa: Dict[str, Any], rounds: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    order = session_order(dialog_data)
    target_sessions = set(qa.get("session_id", []))

    history: List[Dict[str, Any]] = []
    for sid in order:
        if sid not in target_sessions:
            continue
        session = next(s for s in dialog_data["multi_session_dialogues"] if s.get("session_id") == sid)
        for d in session.get("dialogues", []):
            rid = d.get("round", "")
            r = rounds.get(rid, {})
            user_text = r.get("user", "")
            assistant_text = r.get("assistant", "")
            images = r.get("images", [])
            if user_text:
                history.append({"role": "user", "text": user_text, "images": images})
            if assistant_text:
                history.append({"role": "assistant", "text": assistant_text, "images": []})
    return history


def to_mcq(question: str) -> str:
    return (
        f"{question}\n"
        "Choose the best option.\n"
        "A) Correct answer exactly matches the key fact.\n"
        "B) Partially correct but misses key detail.\n"
        "C) Incorrect.\n"
        "Reply with only A, B, or C."
    )


def extract_choice(text: str) -> str:
    t = text.strip().upper()
    if t in {"A", "B", "C"}:
        return t
    m = re.search(r"\b([ABC])\b", t)
    return m.group(1) if m else "INVALID"


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_open(pred: str, gt: str) -> Tuple[bool, bool]:
    p = normalize(pred)
    g = normalize(gt)
    exact = p == g and p != ""
    contains = g in p if g else False
    return exact, contains


def get_point_keywords() -> Dict[str, List[str]]:
    return {
        "FR": ["glass", "contour", "bottle", "glass contour bottle"],
        "VS": ["orange", "pink"],
        "MR": ["coca-cola", "coca cola", "red background", "solid red"],
        "CD": ["no", "not", "analyzing", "analysis", "testing", "extraction"],
    }


def score_open_soft(point: str, pred: str, gt: str) -> Tuple[List[str], float]:
    p = normalize(pred)
    g = normalize(gt)
    kws = get_point_keywords().get(point, [])
    hits = [kw for kw in kws if kw in p]
    base = len(hits) / len(kws) if kws else 0.0
    # small bonus if exact GT phrase appears
    bonus = 0.2 if g and g in p else 0.0
    score = min(1.0, base + bonus)
    return hits, score


def get_qas(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("human-annotated QAs", "human_annotated_qas", "qas"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    config_dir = resolve_config_path(args.config).parent
    dialog_json = resolve_dataset_path(str(cfg["dataset"]["dialog_json"]), config_dir)
    image_root_raw = str(cfg["dataset"].get("image_root", "")).strip()
    image_root = resolve_dataset_path(image_root_raw, config_dir) if image_root_raw else None
    output_json = resolve_dataset_path(str(cfg["eval"]["output_json"]), config_dir)
    mode = cfg["eval"].get("mode", "open")

    data = load_json(dialog_json)
    rounds = build_rounds(data, dialog_json, image_root)

    router = QwenLocalRouter(
        model_path=cfg["model"]["model_path"],
        max_new_tokens=int(cfg["model"].get("max_new_tokens", 128)),
    )

    qas = get_qas(data)
    results: List[Dict[str, Any]] = []
    point_counter: Dict[str, Dict[str, float]] = {}
    for i, qa in enumerate(qas, start=1):
        question = qa.get("question", "")
        gt = qa.get("answer", "")
        history = build_history_for_qa(data, qa, rounds)
        print(f"[INFO] QA {i}/{len(qas)} point={qa.get('point')} history_turns={len(history)}")

        if mode in {"open", "both"}:
            t0 = dt.datetime.now()
            pred = router.answer(history, question)
            dt_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)
            exact, contains = score_open(pred, gt)
            hits, soft = score_open_soft(qa.get("point", ""), pred, gt)
            results.append(
                {
                    "idx": i,
                    "point": qa.get("point"),
                    "mode": "open",
                    "question": question,
                    "pred": pred,
                    "gt": gt,
                    "exact_match": exact,
                    "contains_gt": contains,
                    "keyword_hits": hits,
                    "open_soft_score": soft,
                    "latency_ms": dt_ms,
                }
            )
            print(
                f"[OPEN][{i}] exact={exact} contains={contains} "
                f"soft={soft:.2f} latency_ms={dt_ms}"
            )
            pt = str(qa.get("point", "UNK"))
            bucket = point_counter.setdefault(pt, {"n": 0.0, "soft_sum": 0.0})
            bucket["n"] += 1
            bucket["soft_sum"] += soft

        if mode in {"mcq", "both"}:
            mcq_q = to_mcq(question)
            t0 = dt.datetime.now()
            pred_raw = router.answer(history, mcq_q)
            dt_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)
            choice = extract_choice(pred_raw)
            results.append(
                {
                    "idx": i,
                    "point": qa.get("point"),
                    "mode": "mcq",
                    "question": mcq_q,
                    "pred_raw": pred_raw,
                    "pred_choice": choice,
                    "valid_choice": choice in {"A", "B", "C"},
                    "gt": gt,
                    "latency_ms": dt_ms,
                }
            )
            print(f"[MCQ][{i}] choice={choice} latency_ms={dt_ms}")

    summary: Dict[str, Any] = {}
    if results:
        open_rows = [r for r in results if r.get("mode") == "open"]
        mcq_rows = [r for r in results if r.get("mode") == "mcq"]
        if open_rows:
            summary["open_count"] = len(open_rows)
            summary["open_exact_rate"] = sum(1 for r in open_rows if r["exact_match"]) / len(open_rows)
            summary["open_contains_rate"] = sum(1 for r in open_rows if r["contains_gt"]) / len(open_rows)
            summary["open_soft_avg"] = sum(float(r.get("open_soft_score", 0.0)) for r in open_rows) / len(
                open_rows
            )
        if mcq_rows:
            summary["mcq_count"] = len(mcq_rows)
            summary["mcq_valid_rate"] = sum(1 for r in mcq_rows if r["valid_choice"]) / len(mcq_rows)

        if point_counter:
            summary["point_soft_avg"] = {
                k: (v["soft_sum"] / v["n"] if v["n"] else 0.0) for k, v in point_counter.items()
            }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dialog_json": str(dialog_json),
        "model_path": cfg["model"]["model_path"],
        "mode": mode,
        "num_qas": len(qas),
        "summary": summary,
        "results": results,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved: {output_json}")


if __name__ == "__main__":
    main()

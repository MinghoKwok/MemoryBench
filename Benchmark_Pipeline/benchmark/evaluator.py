import re
from typing import Any, Dict, List, Tuple


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
    match = re.search(r"\b([ABC])\b", t)
    return match.group(1) if match else "INVALID"


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
    keywords = get_point_keywords().get(point, [])
    hits = [kw for kw in keywords if kw in p]
    base = len(hits) / len(keywords) if keywords else 0.0
    bonus = 0.2 if g and g in p else 0.0
    score = min(1.0, base + bonus)
    return hits, score


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    point_counter: Dict[str, Dict[str, float]] = {}
    open_rows = [r for r in results if r.get("mode") == "open"]
    mcq_rows = [r for r in results if r.get("mode") == "mcq"]

    if open_rows:
        summary["open_count"] = len(open_rows)
        summary["open_exact_rate"] = sum(1 for r in open_rows if r["exact_match"]) / len(open_rows)
        summary["open_contains_rate"] = sum(1 for r in open_rows if r["contains_gt"]) / len(open_rows)
        summary["open_soft_avg"] = sum(float(r.get("open_soft_score", 0.0)) for r in open_rows) / len(
            open_rows
        )
        for row in open_rows:
            point = str(row.get("point", "UNK"))
            bucket = point_counter.setdefault(point, {"n": 0.0, "soft_sum": 0.0})
            bucket["n"] += 1
            bucket["soft_sum"] += float(row.get("open_soft_score", 0.0))

    if mcq_rows:
        summary["mcq_count"] = len(mcq_rows)
        summary["mcq_valid_rate"] = sum(1 for r in mcq_rows if r["valid_choice"]) / len(mcq_rows)

    if point_counter:
        summary["point_soft_avg"] = {
            point: (bucket["soft_sum"] / bucket["n"] if bucket["n"] else 0.0)
            for point, bucket in point_counter.items()
        }

    return summary

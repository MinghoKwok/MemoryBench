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


def point_axes(point: Any) -> Tuple[List[str], List[str]]:
    if isinstance(point, list) and len(point) >= 2:
        x_axis = [str(v) for v in point[0] if str(v).strip()]
        y_axis = [str(v) for v in point[1] if str(v).strip()]
        return x_axis, y_axis
    if isinstance(point, str) and point.strip():
        return [point.strip()], []
    return [], []


def point_signature(point: Any) -> str:
    x_axis, y_axis = point_axes(point)
    if not x_axis and not y_axis:
        return "UNK"
    if x_axis and y_axis:
        return f"{'+'.join(x_axis)}|{'+'.join(y_axis)}"
    if x_axis:
        return "+".join(x_axis)
    return "+".join(y_axis)


def get_point_keywords() -> Dict[str, List[str]]:
    return {
        "FR": ["glass", "contour", "bottle", "glass contour bottle"],
        "VS": ["orange", "pink"],
        "MR": ["coca-cola", "coca cola", "red background", "solid red"],
        "CD": ["no", "not", "analyzing", "analysis", "testing", "extraction"],
    }


def score_open_soft(point: Any, pred: str, gt: str) -> Tuple[List[str], float]:
    p = normalize(pred)
    g = normalize(gt)
    keywords = get_point_keywords().get(point if isinstance(point, str) else "", [])
    hits = [kw for kw in keywords if kw in p]
    base = len(hits) / len(keywords) if keywords else 0.0
    bonus = 0.2 if g and g in p else 0.0
    score = min(1.0, base + bonus)
    return hits, score


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    point_counter: Dict[str, Dict[str, float]] = {}
    x_counter: Dict[str, Dict[str, float]] = {}
    y_counter: Dict[str, Dict[str, float]] = {}
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
            point = row.get("point")
            bucket = point_counter.setdefault(point_signature(point), {"n": 0.0, "soft_sum": 0.0})
            bucket["n"] += 1
            bucket["soft_sum"] += float(row.get("open_soft_score", 0.0))
            x_axis, y_axis = point_axes(point)
            for x_label in x_axis:
                x_bucket = x_counter.setdefault(x_label, {"n": 0.0, "soft_sum": 0.0})
                x_bucket["n"] += 1
                x_bucket["soft_sum"] += float(row.get("open_soft_score", 0.0))
            for y_label in y_axis:
                y_bucket = y_counter.setdefault(y_label, {"n": 0.0, "soft_sum": 0.0})
                y_bucket["n"] += 1
                y_bucket["soft_sum"] += float(row.get("open_soft_score", 0.0))

    if mcq_rows:
        summary["mcq_count"] = len(mcq_rows)
        summary["mcq_valid_rate"] = sum(1 for r in mcq_rows if r["valid_choice"]) / len(mcq_rows)

    if point_counter:
        summary["point_soft_avg"] = {
            point: (bucket["soft_sum"] / bucket["n"] if bucket["n"] else 0.0)
            for point, bucket in point_counter.items()
        }
    if x_counter:
        summary["x_soft_avg"] = {
            label: (bucket["soft_sum"] / bucket["n"] if bucket["n"] else 0.0)
            for label, bucket in x_counter.items()
        }
    if y_counter:
        summary["y_soft_avg"] = {
            label: (bucket["soft_sum"] / bucket["n"] if bucket["n"] else 0.0)
            for label, bucket in y_counter.items()
        }

    return summary

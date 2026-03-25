import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Tuple

from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

_stemmer = PorterStemmer()
_ARTICLES = re.compile(r"\b(a|an|the)\s+(?=\w)")


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
    text = str(s).lower()
    text = re.sub(r"(?<=\d)\.(?=\d)", "DOTPLACEHOLDER", text)
    text = text.replace("_", "UNDERSCOREPLACEHOLDER")
    text = _ARTICLES.sub(" ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = text.replace("DOTPLACEHOLDER", ".").replace("UNDERSCOREPLACEHOLDER", "_")
    return " ".join(text.split())


def normalized_tokens(s: str) -> List[str]:
    return normalize(s).split()


def score_open(pred: str, gt: str) -> Tuple[bool, bool]:
    p = normalize(pred)
    g = normalize(gt)
    exact = p == g and p != ""
    contains = g in p if g else False
    return exact, contains


def f1_score(pred: str, gt: str) -> float:
    pred_tokens = [_stemmer.stem(tok) for tok in normalized_tokens(pred)]
    gt_tokens = [_stemmer.stem(tok) for tok in normalized_tokens(gt)]
    if not pred_tokens or not gt_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(pred: str, gt: str, weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)) -> float:
    pred_tokens = normalized_tokens(pred)
    gt_tokens = normalized_tokens(gt)
    if not pred_tokens or not gt_tokens:
        return 0.0
    return float(
        sentence_bleu(
            [gt_tokens],
            pred_tokens,
            weights=weights,
            smoothing_function=SmoothingFunction().method1,
        )
    )


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


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    point_counter: Dict[str, Dict[str, float]] = {}
    x_counter: Dict[str, Dict[str, float]] = {}
    y_counter: Dict[str, Dict[str, float]] = {}
    open_rows = [r for r in results if r.get("mode") == "open"]
    mcq_rows = [r for r in results if r.get("mode") == "mcq"]

    def _metric_bucket() -> Dict[str, float]:
        return {
            "n": 0.0,
            "em_sum": 0.0,
            "contains_sum": 0.0,
            "f1_sum": 0.0,
            "bleu_1_sum": 0.0,
            "bleu_2_sum": 0.0,
        }

    def _bucket_metrics(bucket: Dict[str, float]) -> Dict[str, float]:
        n = bucket["n"]
        if not n:
            return {}
        return {
            "open_exact_rate": bucket["em_sum"] / n,
            "open_contains_rate": bucket["contains_sum"] / n,
            "open_f1_avg": bucket["f1_sum"] / n,
            "open_bleu_1_avg": bucket["bleu_1_sum"] / n,
            "open_bleu_2_avg": bucket["bleu_2_sum"] / n,
        }

    if open_rows:
        summary["open_count"] = len(open_rows)
        summary["open_exact_rate"] = sum(1 for r in open_rows if r["exact_match"]) / len(open_rows)
        summary["open_contains_rate"] = sum(1 for r in open_rows if r["contains_gt"]) / len(open_rows)
        summary["open_f1_avg"] = sum(float(r.get("f1", 0.0)) for r in open_rows) / len(open_rows)
        summary["open_bleu_1_avg"] = sum(float(r.get("bleu_1", 0.0)) for r in open_rows) / len(open_rows)
        summary["open_bleu_2_avg"] = sum(float(r.get("bleu_2", 0.0)) for r in open_rows) / len(open_rows)
        for row in open_rows:
            point = row.get("point")
            bucket = point_counter.setdefault(point_signature(point), _metric_bucket())
            bucket["n"] += 1
            bucket["em_sum"] += 1.0 if row.get("exact_match") else 0.0
            bucket["contains_sum"] += 1.0 if row.get("contains_gt") else 0.0
            bucket["f1_sum"] += float(row.get("f1", 0.0))
            bucket["bleu_1_sum"] += float(row.get("bleu_1", 0.0))
            bucket["bleu_2_sum"] += float(row.get("bleu_2", 0.0))
            x_axis, y_axis = point_axes(point)
            for x_label in x_axis:
                x_bucket = x_counter.setdefault(x_label, _metric_bucket())
                x_bucket["n"] += 1
                x_bucket["em_sum"] += 1.0 if row.get("exact_match") else 0.0
                x_bucket["contains_sum"] += 1.0 if row.get("contains_gt") else 0.0
                x_bucket["f1_sum"] += float(row.get("f1", 0.0))
                x_bucket["bleu_1_sum"] += float(row.get("bleu_1", 0.0))
                x_bucket["bleu_2_sum"] += float(row.get("bleu_2", 0.0))
            for y_label in y_axis:
                y_bucket = y_counter.setdefault(y_label, _metric_bucket())
                y_bucket["n"] += 1
                y_bucket["em_sum"] += 1.0 if row.get("exact_match") else 0.0
                y_bucket["contains_sum"] += 1.0 if row.get("contains_gt") else 0.0
                y_bucket["f1_sum"] += float(row.get("f1", 0.0))
                y_bucket["bleu_1_sum"] += float(row.get("bleu_1", 0.0))
                y_bucket["bleu_2_sum"] += float(row.get("bleu_2", 0.0))

    if mcq_rows:
        summary["mcq_count"] = len(mcq_rows)
        summary["mcq_valid_rate"] = sum(1 for r in mcq_rows if r["valid_choice"]) / len(mcq_rows)

    if point_counter:
        summary["point_metrics"] = {point: _bucket_metrics(bucket) for point, bucket in point_counter.items()}
    if x_counter:
        summary["x_metrics"] = {label: _bucket_metrics(bucket) for label, bucket in x_counter.items()}
    if y_counter:
        summary["y_metrics"] = {label: _bucket_metrics(bucket) for label, bucket in y_counter.items()}

    return summary

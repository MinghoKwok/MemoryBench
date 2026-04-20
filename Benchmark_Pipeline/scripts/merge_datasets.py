"""Merge two MemEye dialog JSONs into a single dataset.

Usage:
    python scripts/merge_datasets.py \
        --inputs A.json B.json \
        --output merged.json \
        --name "Merged_Task_Name" \
        --qa-prefixes ANIM COMIC \
        --date-base 2025-01-01 \
        --date-gap 3
"""
import argparse
import copy
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def _is_iso_date(d: str) -> bool:
    return bool(d) and bool(re.match(r"^\d{4}-\d{2}-\d{2}", d))


def _normalize_dates(
    sessions: list[dict],
    base: datetime,
    gap_days: int,
) -> list[dict]:
    """Assign ISO dates to sessions that lack them, preserving original order.

    Sessions that already have ISO dates are rebased so the earliest one maps
    to *base* and the relative spacing is preserved.  Sessions with non-ISO
    labels (e.g. episode names) get evenly-spaced synthetic dates starting
    from *base*.
    """
    out = []
    for i, s in enumerate(sessions):
        s = copy.deepcopy(s)
        d = s.get("date", "")
        if _is_iso_date(d):
            # Will be rebased below
            out.append(s)
        else:
            # Synthetic date
            s["date"] = (base + timedelta(days=i * gap_days)).strftime("%Y-%m-%d")
            out.append(s)
    return out


def _rebase_iso_dates(
    sessions: list[dict],
    target_start: datetime,
) -> list[dict]:
    """Shift all ISO dates so the earliest becomes *target_start*,
    preserving relative spacing between sessions."""
    iso_dates = []
    for s in sessions:
        d = s.get("date", "")
        if _is_iso_date(d):
            iso_dates.append(datetime.strptime(d[:10], "%Y-%m-%d"))

    if not iso_dates:
        return sessions

    earliest = min(iso_dates)
    offset = target_start - earliest

    out = []
    for s in sessions:
        s = copy.deepcopy(s)
        d = s.get("date", "")
        if _is_iso_date(d):
            old_dt = datetime.strptime(d[:10], "%Y-%m-%d")
            s["date"] = (old_dt + offset).strftime("%Y-%m-%d")
        out.append(s)
    return out


def merge_datasets(
    paths: list[str],
    output: str,
    name: str,
    qa_prefixes: list[str],
    date_base: str = "2025-01-01",
    date_gap: int = 3,
) -> None:
    datasets = []
    for p in paths:
        with open(p) as f:
            datasets.append(json.load(f))

    if len(qa_prefixes) != len(datasets):
        raise ValueError(
            f"Need {len(datasets)} QA prefixes, got {len(qa_prefixes)}"
        )

    base_dt = datetime.strptime(date_base, "%Y-%m-%d")

    # --- Normalize dates per dataset, then concatenate ---
    all_sessions = []
    cursor = base_dt
    for ds in datasets:
        raw_sessions = ds.get("multi_session_dialogues", [])
        has_iso = any(_is_iso_date(s.get("date", "")) for s in raw_sessions)

        if has_iso:
            # Rebase existing ISO dates so earliest = cursor
            normalized = _rebase_iso_dates(raw_sessions, cursor)
        else:
            # Generate synthetic dates starting from cursor
            normalized = _normalize_dates(raw_sessions, cursor, date_gap)

        all_sessions.extend(normalized)

        # Advance cursor past this dataset's last date + gap
        last_dates = []
        for s in normalized:
            d = s.get("date", "")
            if _is_iso_date(d):
                last_dates.append(datetime.strptime(d[:10], "%Y-%m-%d"))
        if last_dates:
            cursor = max(last_dates) + timedelta(days=date_gap)

    # --- Merge QAs with prefixed question_ids ---
    all_qas = []
    for ds, prefix in zip(datasets, qa_prefixes):
        for qa in ds.get("human-annotated QAs", []):
            new_qa = copy.deepcopy(qa)
            old_id = new_qa.get("question_id", "")
            new_qa["question_id"] = f"{prefix}_{old_id}"
            all_qas.append(new_qa)

    # --- Merge character profiles ---
    profiles = [ds.get("character_profile", {}) for ds in datasets]
    merged_profile = {}
    names = [p.get("name", "") for p in profiles if p.get("name")]
    roles = [p.get("role", "") for p in profiles if p.get("role")]
    if names:
        merged_profile["name"] = " & ".join(names)
    if roles:
        merged_profile["role"] = " | ".join(roles)
    for p in profiles:
        for k, v in p.items():
            if k not in merged_profile and v:
                merged_profile[k] = v

    merged = {
        "task_name": name,
        "character_profile": merged_profile,
        "multi_session_dialogues": all_sessions,
        "human-annotated QAs": all_qas,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"Merged {len(datasets)} datasets:")
    for p, prefix in zip(paths, qa_prefixes):
        print(f"  {prefix}: {p}")
    print(f"Sessions: {len(all_sessions)}")
    print(f"QAs: {len(all_qas)}")
    dates = [s["date"] for s in all_sessions if _is_iso_date(s.get("date", ""))]
    if dates:
        print(f"Date range: {min(dates)} → {max(dates)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--qa-prefixes", nargs="+", required=True)
    parser.add_argument("--date-base", default="2025-01-01")
    parser.add_argument("--date-gap", type=int, default=3)
    args = parser.parse_args()
    merge_datasets(
        args.inputs, args.output, args.name, args.qa_prefixes,
        args.date_base, args.date_gap,
    )

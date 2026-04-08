#!/usr/bin/env python3
"""Comprehensive analysis of MemoryBench experiment results."""
import json, os, glob
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path("runs")
# Collect latest run per (task, model, method)
def collect_latest_runs():
    runs = {}
    for metrics_path in RUNS_DIR.rglob("metrics.json"):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        pred_path = run_dir / "predictions.jsonl"
        if not config_path.exists() or not pred_path.exists():
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        task = cfg.get("task", {}).get("name", "unknown")
        model = cfg.get("model", {}).get("name", "unknown")
        method = cfg.get("method", {}).get("name", "unknown")
        timestamp = run_dir.name[:15]  # e.g. 20260331_161724
        key = (task, model, method)
        if key not in runs or timestamp > runs[key]["timestamp"]:
            runs[key] = {
                "timestamp": timestamp,
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "pred_path": str(pred_path),
                "task": task, "model": model, "method": method,
            }
    return runs

def load_predictions(pred_path):
    preds = []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return preds

def main():
    runs = collect_latest_runs()
    print(f"Found {len(runs)} unique (task, model, method) combinations\n")

    # Group by task
    tasks = defaultdict(dict)
    for (task, model, method), info in runs.items():
        tasks[task][method] = info

    # ============================================================
    # 1. OVERALL METRICS TABLE PER TASK
    # ============================================================
    print("=" * 100)
    print("1. OVERALL METRICS BY TASK × METHOD (F1 / EM / contains_gt)")
    print("=" * 100)

    METHODS_ORDER = ["full_context_multimodal", "full_context_text_only", "semantic_rag_text_only", "semantic_rag_multimodal", "m2a", "mma"]

    for task_name in sorted(tasks.keys()):
        methods = tasks[task_name]
        print(f"\n### {task_name}")
        print(f"  {'Method':<30} {'F1':>8} {'EM':>8} {'contains_gt':>12} {'BLEU-1':>8} {'#QAs':>6}")
        print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*4}")
        for m in METHODS_ORDER:
            if m not in methods:
                continue
            with open(methods[m]["metrics_path"]) as f:
                met = json.load(f)
            s = met["summary"]["overall"]
            n = met.get("num_qas_run", met.get("num_qas", "?"))
            print(f"  {m:<30} {s['f1']:>8.4f} {s['em']:>8.4f} {s['contains_gt']:>12.4f} {s['bleu_1']:>8.4f} {n:>6}")

    # ============================================================
    # 2. METRICS BY MEMEYE CELL (X × Y) PER TASK × METHOD
    # ============================================================
    print("\n" + "=" * 100)
    print("2. F1 BREAKDOWN BY MEMEYE COORDINATES (X-axis × Y-axis)")
    print("=" * 100)

    for task_name in sorted(tasks.keys()):
        methods = tasks[task_name]
        print(f"\n### {task_name}")

        # Collect all cells across methods
        all_x = set()
        all_y = set()
        for m_name, info in methods.items():
            with open(info["metrics_path"]) as f:
                met = json.load(f)
            for cell_key in met["summary"].get("by_cell", {}):
                parts = cell_key.split("_")
                all_x.add(parts[0])
                all_y.add(parts[1])

        xs = sorted(all_x)
        ys = sorted(all_y)

        # Print by_x
        print(f"\n  By X-axis (Visual Demands):")
        header = f"  {'Method':<25}"
        for x in xs:
            header += f" {x:>8}"
        print(header)
        for m in METHODS_ORDER:
            if m not in methods:
                continue
            with open(methods[m]["metrics_path"]) as f:
                met = json.load(f)
            row = f"  {m:<25}"
            for x in xs:
                val = met["summary"].get("by_x", {}).get(x, {}).get("f1")
                row += f" {val:>8.3f}" if val is not None else f" {'N/A':>8}"
            print(row)

        # Print by_y
        print(f"\n  By Y-axis (Reasoning Demands):")
        header = f"  {'Method':<25}"
        for y in ys:
            header += f" {y:>8}"
        print(header)
        for m in METHODS_ORDER:
            if m not in methods:
                continue
            with open(methods[m]["metrics_path"]) as f:
                met = json.load(f)
            row = f"  {m:<25}"
            for y in ys:
                val = met["summary"].get("by_y", {}).get(y, {}).get("f1")
                row += f" {val:>8.3f}" if val is not None else f" {'N/A':>8}"
            print(row)

    # ============================================================
    # 3. PER-QUESTION CROSS-METHOD ANALYSIS
    # ============================================================
    print("\n" + "=" * 100)
    print("3. PER-QUESTION ANALYSIS: Method Agreement / Disagreement")
    print("=" * 100)

    for task_name in sorted(tasks.keys()):
        methods = tasks[task_name]
        if len(methods) < 2:
            continue

        print(f"\n### {task_name}")

        # Load all predictions indexed by question idx
        method_preds = {}
        for m_name, info in methods.items():
            preds = load_predictions(info["pred_path"])
            method_preds[m_name] = {p["idx"]: p for p in preds}

        available_methods = [m for m in METHODS_ORDER if m in method_preds]

        # Find all question idxs
        all_idxs = set()
        for mp in method_preds.values():
            all_idxs.update(mp.keys())

        # Classify questions
        all_correct = []  # All methods got it right (f1 > 0.5)
        all_wrong = []    # All methods got it wrong (f1 < 0.2)
        method_diverge = []  # Some right, some wrong

        for idx in sorted(all_idxs):
            scores = {}
            for m in available_methods:
                if idx in method_preds[m]:
                    scores[m] = method_preds[m][idx]

            f1s = {m: s.get("f1", 0) for m, s in scores.items()}
            high = [m for m, f in f1s.items() if f >= 0.5]
            low = [m for m, f in f1s.items() if f < 0.2]

            q = list(scores.values())[0]  # representative
            entry = {
                "idx": idx,
                "question": q.get("question", ""),
                "gt": q.get("gt", ""),
                "point": q.get("point", []),
                "clue": q.get("clue_rounds", q.get("clue", "")),
                "f1s": f1s,
                "preds": {m: scores[m].get("pred", "") for m in available_methods if m in scores},
            }

            if len(high) == len(scores):
                all_correct.append(entry)
            elif len(low) == len(scores):
                all_wrong.append(entry)
            elif len(high) > 0 and len(low) > 0:
                method_diverge.append(entry)

        total = len(all_idxs)
        print(f"  Total QAs: {total}")
        print(f"  All methods correct (F1≥0.5): {len(all_correct)} ({100*len(all_correct)/total:.0f}%)")
        print(f"  All methods wrong (F1<0.2):   {len(all_wrong)} ({100*len(all_wrong)/total:.0f}%)")
        print(f"  Methods diverge:              {len(method_diverge)} ({100*len(method_diverge)/total:.0f}%)")

        # ============================================================
        # 4. CASE STUDIES: Divergent questions
        # ============================================================
        if method_diverge:
            print(f"\n  --- CASE STUDIES: Method Divergence (up to 5) ---")
            for entry in method_diverge[:5]:
                point_str = str(entry["point"])
                print(f"\n  Q{entry['idx']} [{point_str}]: {entry['question'][:120]}")
                print(f"  GT: {entry['gt'][:120]}")
                print(f"  Clue: {entry['clue']}")
                for m in available_methods:
                    if m in entry["f1s"]:
                        f1 = entry["f1s"][m]
                        pred = entry["preds"].get(m, "N/A")[:100]
                        marker = "✓" if f1 >= 0.5 else "✗"
                        print(f"    {marker} {m:<28} F1={f1:.3f}  → {pred}")

        # All-wrong case studies
        if all_wrong:
            print(f"\n  --- CASE STUDIES: Universally Failed (up to 5) ---")
            for entry in all_wrong[:5]:
                point_str = str(entry["point"])
                print(f"\n  Q{entry['idx']} [{point_str}]: {entry['question'][:120]}")
                print(f"  GT: {entry['gt'][:120]}")
                print(f"  Clue: {entry['clue']}")
                for m in available_methods:
                    if m in entry["f1s"]:
                        pred = entry["preds"].get(m, "N/A")[:100]
                        print(f"    ✗ {m:<28} F1={entry['f1s'][m]:.3f}  → {pred}")

    # ============================================================
    # 5. METHOD STRENGTH ANALYSIS: Which MemEye cells favor which method?
    # ============================================================
    print("\n" + "=" * 100)
    print("4. METHOD STRENGTH HEATMAP: Best method per MemEye cell (across all tasks)")
    print("=" * 100)

    cell_method_f1 = defaultdict(lambda: defaultdict(list))

    for task_name in sorted(tasks.keys()):
        methods = tasks[task_name]
        for m_name, info in methods.items():
            preds = load_predictions(info["pred_path"])
            for p in preds:
                point = p.get("point", [])
                if len(point) >= 2:
                    xs = point[0] if isinstance(point[0], list) else [point[0]]
                    ys = point[1] if isinstance(point[1], list) else [point[1]]
                    for x in xs:
                        for y in ys:
                            cell = f"{x}_{y}"
                            cell_method_f1[cell][m_name].append(p.get("f1", 0))

    # Print heatmap
    all_cells_sorted = sorted(cell_method_f1.keys())
    all_methods_seen = sorted(set(m for cell in cell_method_f1 for m in cell_method_f1[cell]))

    print(f"\n  {'Cell':<10}", end="")
    for m in METHODS_ORDER:
        if m in all_methods_seen:
            print(f" {m[:18]:>18}", end="")
    print(f" {'Best':>20} {'N':>5}")

    for cell in all_cells_sorted:
        print(f"  {cell:<10}", end="")
        best_m, best_f1 = "", 0
        for m in METHODS_ORDER:
            if m in all_methods_seen and m in cell_method_f1[cell]:
                avg = sum(cell_method_f1[cell][m]) / len(cell_method_f1[cell][m])
                print(f" {avg:>18.3f}", end="")
                if avg > best_f1:
                    best_f1, best_m = avg, m
            else:
                print(f" {'—':>18}", end="")
        n = sum(len(v) for v in cell_method_f1[cell].values()) // max(len(cell_method_f1[cell]), 1)
        print(f" {best_m:>20} {n:>5}")

    # ============================================================
    # 6. RETRIEVAL FAILURE ANALYSIS for RAG methods
    # ============================================================
    print("\n" + "=" * 100)
    print("5. RAG RETRIEVAL HIT/MISS ANALYSIS")
    print("=" * 100)

    for task_name in sorted(tasks.keys()):
        methods = tasks[task_name]
        for m_name in ["semantic_rag_text_only", "semantic_rag_multimodal"]:
            if m_name not in methods:
                continue
            preds = load_predictions(methods[m_name]["pred_path"])

            # Check if predictions have retrieval info
            has_history = any("history_turns" in p for p in preds)
            if not has_history:
                continue

            high_history_high_f1 = []
            low_history = []

            for p in preds:
                ht = p.get("history_turns", 0)
                f1 = p.get("f1", 0)
                if ht == 0:
                    low_history.append(p)
                elif f1 >= 0.5:
                    high_history_high_f1.append(p)

            total = len(preds)
            if total == 0:
                continue
            avg_f1 = sum(p.get("f1", 0) for p in preds) / total
            zero_hist = len([p for p in preds if p.get("history_turns", 0) == 0])

            print(f"\n  {task_name} / {m_name}:")
            print(f"    Avg F1: {avg_f1:.3f}, Zero-history QAs: {zero_hist}/{total}")

            # Show distribution of history_turns
            hist_counts = defaultdict(int)
            for p in preds:
                ht = p.get("history_turns", 0)
                hist_counts[ht] += 1
            print(f"    History turns distribution: {dict(sorted(hist_counts.items()))}")

    # ============================================================
    # 7. SUMMARY: Win/Loss per method across all tasks
    # ============================================================
    print("\n" + "=" * 100)
    print("6. WIN COUNT: How often each method achieves highest F1 per question")
    print("=" * 100)

    method_wins = defaultdict(int)
    method_total = defaultdict(int)
    method_f1_sum = defaultdict(float)

    for task_name in sorted(tasks.keys()):
        methods = tasks[task_name]
        method_preds = {}
        for m_name, info in methods.items():
            preds = load_predictions(info["pred_path"])
            method_preds[m_name] = {p["idx"]: p for p in preds}

        all_idxs = set()
        for mp in method_preds.values():
            all_idxs.update(mp.keys())

        for idx in sorted(all_idxs):
            best_f1 = -1
            best_methods = []
            for m in method_preds:
                if idx in method_preds[m]:
                    f1 = method_preds[m][idx].get("f1", 0)
                    method_f1_sum[m] += f1
                    method_total[m] += 1
                    if f1 > best_f1:
                        best_f1 = f1
                        best_methods = [m]
                    elif f1 == best_f1:
                        best_methods.append(m)
            for m in best_methods:
                method_wins[m] += 1

    print(f"\n  {'Method':<30} {'Wins':>6} {'Total':>6} {'Win%':>8} {'Avg F1':>8}")
    for m in METHODS_ORDER:
        if m in method_wins:
            w = method_wins[m]
            t = method_total[m]
            avg = method_f1_sum[m] / t if t > 0 else 0
            print(f"  {m:<30} {w:>6} {t:>6} {100*w/t:>7.1f}% {avg:>8.4f}")

if __name__ == "__main__":
    main()

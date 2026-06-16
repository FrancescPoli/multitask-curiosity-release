"""
Extract Performance Metrics
============================

Iterates over all runs in a sweep directory and extracts training performance
metrics from each run's `run_curriculum.csv`.

Metrics per run:
  - mean_accuracy:    Mean final EMA accuracy across all training tasks
  - fraction_solved:  Fraction of tasks reaching >= 80% accuracy
  - mean_speed:       Mean steps to reach 80% (NaN if no task solved)
  - n_tasks_solved:   Count of tasks reaching >= 80%
  - n_tasks_total:    Total number of training tasks
  - switch_rate:      Task switches per 1000 steps (foraging exploration)
  - entropy:          Shannon entropy of task distribution (foraging diversity)

Output:
  analysis/Performance_analysis/Results/performance_metrics.csv

Usage:
  python analysis/Performance_analysis/extract_performance.py \
      --sweep_dir /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6
"""

import csv
import argparse
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy


# ── Helpers ──────────────────────────────────────────────────────────

def build_per_task_ema(csv_path: Path, alpha: float = 0.1):
    """
    Read run_curriculum.csv and build per-task EMA accuracy histories.
    Lightweight reimplementation (no torch dependency).

    Returns dict {task_name: list[float]}  — one EMA value per training step
    on that task.
    """
    ema = {}        # task -> current EMA value
    histories = {}  # task -> [ema_val, ...]

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task")
            acc_str = row.get("acc") or row.get("ema")
            if task is None or acc_str is None:
                continue
            try:
                acc = float(acc_str)
            except ValueError:
                continue

            old = ema.get(task, acc)
            new = (1.0 - alpha) * old + alpha * acc
            ema[task] = new
            histories.setdefault(task, []).append(new)

    return histories


def compute_final_accs(histories, window: int = 100):
    """Mean of last `window` EMA values per task."""
    accs = {}
    for task, vals in histories.items():
        if len(vals) >= window:
            accs[task] = float(np.mean(vals[-window:]))
        elif vals:
            accs[task] = float(np.mean(vals))
    return accs


def compute_speeds(histories, threshold: float = 0.8):
    """First EMA step index where accuracy >= threshold, per task."""
    speeds = {}
    for task, vals in histories.items():
        arr = np.array(vals)
        idxs = np.where(arr >= threshold)[0]
        speeds[task] = int(idxs[0]) if len(idxs) > 0 else np.nan
    return speeds


def compute_foraging_metrics(csv_path: Path):
    """Switch rate and entropy from task sequence."""
    try:
        df = pd.read_csv(csv_path, usecols=["task"])
    except Exception:
        return {"switch_rate": np.nan, "entropy": np.nan}

    tasks = df["task"].values
    n = len(tasks)
    if n == 0:
        return {"switch_rate": np.nan, "entropy": np.nan}

    transitions = int(np.sum(tasks[1:] != tasks[:-1]))
    switch_rate = (transitions / n) * 1000

    counts = pd.Series(tasks).value_counts(normalize=True)
    ent = float(entropy(counts))

    return {"switch_rate": switch_rate, "entropy": ent}


def extract_one_run(run_dir: Path, window: int = 100, threshold: float = 0.8):
    """Extract all performance metrics for a single run. Returns dict or None."""
    csv_path = run_dir / "run_curriculum.csv"
    if not csv_path.exists():
        return None

    # Check model completed
    if not (run_dir / "model_last.pt").exists():
        return None

    try:
        histories = build_per_task_ema(csv_path)
    except Exception as e:
        print(f"  [{run_dir.name}] Error reading curriculum: {e}")
        return None

    if not histories:
        return None

    final_accs = compute_final_accs(histories, window)
    speeds = compute_speeds(histories, threshold)
    foraging = compute_foraging_metrics(csv_path)

    all_accs = list(final_accs.values())
    mean_accuracy = float(np.mean(all_accs)) if all_accs else np.nan
    n_solved = sum(1 for a in all_accs if a >= threshold)
    n_total = len(all_accs)
    fraction_solved = n_solved / n_total if n_total > 0 else 0.0

    valid_speeds = [s for s in speeds.values() if not np.isnan(s)]
    mean_speed = float(np.mean(valid_speeds)) if valid_speeds else np.nan

    return {
        "run_id": run_dir.name,
        "mean_accuracy": round(mean_accuracy, 4),
        "fraction_solved": round(fraction_solved, 4),
        "n_tasks_solved": n_solved,
        "n_tasks_total": n_total,
        "mean_speed": round(mean_speed, 1) if not np.isnan(mean_speed) else np.nan,
        "switch_rate": round(foraging["switch_rate"], 2) if not np.isnan(foraging["switch_rate"]) else np.nan,
        "entropy": round(foraging["entropy"], 4) if not np.isnan(foraging["entropy"]) else np.nan,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract performance metrics from sweep runs")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Root sweep directory containing run_* folders")
    parser.add_argument("--output", type=str,
                        default="analysis/Performance_analysis/Results/performance_metrics.csv",
                        help="Output CSV path")
    parser.add_argument("--window", type=int, default=100,
                        help="Smoothing window for final accuracy (last N EMA values)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Accuracy threshold for 'solved' (default 0.8)")
    args = parser.parse_args()

    sweep_path = Path(args.sweep_dir)
    output_path = Path(args.output)

    if not sweep_path.exists():
        print(f"Directory not found: {sweep_path}")
        sys.exit(1)

    # Discover runs
    run_dirs = sorted([d for d in sweep_path.iterdir()
                       if d.is_dir() and d.name.startswith("run_")])
    print(f"Found {len(run_dirs)} runs in {sweep_path}")

    # Resume support: skip already-processed run_ids
    processed_ids = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            if "run_id" in existing.columns:
                processed_ids = set(existing["run_id"].unique())
                print(f"Resuming: {len(processed_ids)} already processed.")
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(run_dirs)
    new_count = 0
    columns = ["run_id", "mean_accuracy", "fraction_solved", "n_tasks_solved",
               "n_tasks_total", "mean_speed", "switch_rate", "entropy"]

    for i, run_dir in enumerate(run_dirs):
        if run_dir.name in processed_ids:
            continue

        if i % 10 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {run_dir.name}...", end="\r", flush=True)

        result = extract_one_run(run_dir, args.window, args.threshold)
        if result is None:
            continue

        # Append to CSV
        write_header = not output_path.exists() or output_path.stat().st_size == 0
        row_df = pd.DataFrame([result], columns=columns)
        row_df.to_csv(output_path, mode="a", header=write_header, index=False)
        new_count += 1

    print(f"\nDone. Extracted {new_count} new runs. Total on disk: {len(processed_ids) + new_count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

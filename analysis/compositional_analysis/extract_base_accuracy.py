"""
extract_base_accuracy.py
========================
End-of-training accuracy on the *base abilities* that the held-out compositional
recombinations redeploy. This is the `z_acc` covariate used by the compositional
analysis (analysis/compositional_analysis/pub_compositional_generalisation.R):
the AFT model adjusts each network's held-out solve time for how well it had
actually learned the underlying base task by the end of training.

For each run it reads only ['task', 'mode', 'ema'] from run_curriculum.csv and
takes, per base task, the last train-row `ema` (the precomputed end-of-training
accuracy). One row per run, one column per base task.

Inputs:
  --sweep_dir   sweep directory holding the per-run run_curriculum.csv files
  --runs-from   CSV with a `run_id` (and optionally `cohort`) column listing the
                runs to extract (default: compositional_analysis/Data/compositional_dataset.csv)
Output:
  --output      compositional_analysis/Data/compositional_base_accuracy.csv  (run_id, cohort, <base tasks>)

Promoted from the former root-level tmp_train_acc.py. Run where Z: is visible
(PowerShell on Windows, or on the cluster against the real sweep path).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Base abilities redeployed by the held-out tasks (the non-dm2 `base_task` set in
# pub_compositional_generalisation.R's HELDOUT_MATCHES). Extend if EXCLUDE_DM2=FALSE.
BASE_TASKS = ["poli.dm1", "poli.antidm1", "poli.dlydm1", "poli.antidlydm1",
              "poli.dlyms", "poli.dlynms"]

DEFAULT_SWEEP    = "Z:/fp02/logs/sweep/forage_v5.1/forage_v6"
DEFAULT_RUNS_CSV = "analysis/compositional_analysis/Data/compositional_dataset.csv"
DEFAULT_OUT      = "analysis/compositional_analysis/Data/compositional_base_accuracy.csv"


def per_task_final_acc(curriculum_csv: Path) -> dict | None:
    """Last train-row EMA per task = end-of-training accuracy. None if no train rows."""
    df = pd.read_csv(curriculum_csv, usecols=["task", "mode", "ema"], on_bad_lines="skip")
    tr = df[df["mode"] == "train"]
    if len(tr) == 0:
        return None
    return tr.dropna(subset=["ema"]).groupby("task", sort=False)["ema"].last().to_dict()


def main():
    ap = argparse.ArgumentParser(description="Extract end-of-training base-ability accuracy per run.")
    ap.add_argument("--sweep_dir", default=DEFAULT_SWEEP, help="sweep dir with per-run run_curriculum.csv")
    ap.add_argument("--runs-from", default=DEFAULT_RUNS_CSV, help="CSV listing run_id (+ optional cohort)")
    ap.add_argument("--output", default=DEFAULT_OUT, help="output CSV path")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    runs = pd.read_csv(args.runs_from)
    cols = ["run_id"] + (["cohort"] if "cohort" in runs.columns else [])
    runs = runs[cols].drop_duplicates()
    print(f"{len(runs)} runs to extract from {sweep}", flush=True)

    rows, n_found = [], 0
    for i, r in enumerate(runs.itertuples(index=False)):
        cp = sweep / r.run_id / "run_curriculum.csv"
        if not cp.exists():
            continue
        try:
            accs = per_task_final_acc(cp)
        except Exception as e:                      # transient Z: drop / malformed file
            print(f"  ERR {r.run_id}: {e}", flush=True)
            continue
        if accs is None:
            continue
        row = {"run_id": r.run_id}
        if "cohort" in cols:
            row["cohort"] = r.cohort
        for t in BASE_TASKS:
            row[t] = accs.get(t, np.nan)
        rows.append(row)
        n_found += 1
        print(f"  [{i + 1}/{len(runs)}] {r.run_id}", flush=True)

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nFound curriculum for {n_found}/{len(runs)} runs -> {args.output}", flush=True)
    if "cohort" in out.columns and n_found:
        print("\nMean end-of-training accuracy per base task, by cohort:")
        print(out.groupby("cohort")[BASE_TASKS].mean().round(3).to_string())


if __name__ == "__main__":
    main()

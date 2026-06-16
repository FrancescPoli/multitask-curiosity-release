#!/usr/bin/env python3
"""
Build compositional_dataset.csv
================================
Merges, for the ~110 probed models only:
  1. grand_unified_metrics_v2.csv  -- all existing metrics
  2. compositional_analysis/Data/cohorts/cohort_map.csv -- cohort assignments
  3. probe_compositionality.json  -- per-task probe results

Output: analysis/compositional_analysis/Data/compositional_dataset.csv
        analysis/compositional_analysis/Data/task_metadata.csv
  - One row per probed model (~110 rows)
  - All columns from grand_unified_metrics_v2.csv retained
  - cohort column: "H-100", "C-temp", "C-reg", "C-both"
    (slash-joined if a model appears in multiple cohorts)
  - probe_acc_<task>        : final accuracy after probe training
  - probe_solved_at_<task>  : step at which acc first exceeded 0.9 (NA if never)
  - probe_mean_acc          : mean final_acc across all 34 probed tasks
  - probe_mean_acc_training : mean final_acc on the 20 training tasks only
  - probe_mean_acc_heldout  : mean final_acc on the 14 held-out tasks only
  - probe_frac_solved          : fraction of all tasks solved
  - probe_frac_solved_training : fraction of training tasks solved
  - probe_frac_solved_heldout  : fraction of held-out tasks solved
"""

import json
import pandas as pd
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]   # compositional_analysis -> analysis -> project root
DATA_DIR    = ROOT / "analysis" / "compositional_analysis" / "Data"
CSV_PATH    = ROOT / "analysis" / "grand_unified_metrics_v2.csv"
COHORT_PATH = DATA_DIR / "cohorts" / "cohort_map.csv"
OUT_PATH    = DATA_DIR / "compositional_dataset.csv"

SWEEP_DIR = Path("Z:/fp02/logs/sweep/forage_v5.1/forage_v6")

# ── Task lists (mirrors get_full_test_set / get_quick_test_set) ────────────────
TRAINING_TASKS = [
    # Reach
    'poli.go', 'poli.rtgo', 'poli.dlygo', 'poli.antigo', 'poli.ctxgo',
    'poli.dlyantigo', 'poli.dlyantictxgo', 'poli.dlyctxgo', 'poli.antictxgo', 'poli.rtantictxgo',
    # Decision
    'poli.dm1', 'poli.dm2', 'poli.dlydm1', 'poli.multidm', 'poli.antidm1', 'poli.antidlydm1',
    # Match
    'poli.dlyms', 'poli.dlynms', 'poli.catdlyms', 'poli.catdlynms',
]

HELDOUT_TASKS = [
    # Group 1: Transport ctx → Decision
    'poli.ctxdm1', 'poli.ctxdm2',
    'poli.ctxdlydm1', 'poli.ctxdlydm2',
    'poli.antictxdm1', 'poli.antictxdm2',
    'poli.antictxdlydm1', 'poli.antictxdlydm2',
    # Group 2: Transport anti → Match
    'poli.antidlyms', 'poli.antidlynms',
    # Group 3: Double transport (ctx + anti) → Match
    'poli.antictxdlyms', 'poli.antictxcatdlyms',
]

ALL_TASKS = TRAINING_TASKS + HELDOUT_TASKS


def safe_col(task: str) -> str:
    """'poli.ctxdm1' -> 'poli_ctxdm1'"""
    return task.replace(".", "_")


def load_probe_data(sweep_dir: Path) -> pd.DataFrame:
    json_files = list(sweep_dir.glob("*/probe_compositionality.json"))
    print(f"Found {len(json_files)} probe_compositionality.json files")

    rows = []
    for jf in json_files:
        run_id = jf.parent.name
        try:
            with open(jf) as f:
                res = json.load(f)
        except Exception as e:
            print(f"  SKIP {jf.parent.name}: {e}")
            continue

        row = {"run_id": run_id}
        for task, task_res in res.items():
            row[f"probe_acc_{safe_col(task)}"]       = task_res.get("final_acc")
            row[f"probe_solved_at_{safe_col(task)}"] = task_res.get("solved_at")
        rows.append(row)

    if not rows:
        print(f"  WARNING: No probe files found under {sweep_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Summary columns ───────────────────────────────────────────────────────
    def acc_cols(tasks):
        return [f"probe_acc_{safe_col(t)}" for t in tasks if f"probe_acc_{safe_col(t)}" in df.columns]

    def solved_cols(tasks):
        return [f"probe_solved_at_{safe_col(t)}" for t in tasks if f"probe_solved_at_{safe_col(t)}" in df.columns]

    all_acc     = acc_cols(ALL_TASKS)
    train_acc   = acc_cols(TRAINING_TASKS)
    heldout_acc = acc_cols(HELDOUT_TASKS)

    df["probe_mean_acc"]          = df[all_acc].mean(axis=1)
    df["probe_mean_acc_training"] = df[train_acc].mean(axis=1)
    df["probe_mean_acc_heldout"]  = df[heldout_acc].mean(axis=1)

    n_all     = len(acc_cols(ALL_TASKS))
    n_train   = len(train_acc)
    n_heldout = len(heldout_acc)

    df["probe_frac_solved"]          = df[solved_cols(ALL_TASKS)].notna().sum(axis=1)  / n_all     if n_all     else float("nan")
    df["probe_frac_solved_training"] = df[solved_cols(TRAINING_TASKS)].notna().sum(axis=1) / n_train  if n_train  else float("nan")
    df["probe_frac_solved_heldout"]  = df[solved_cols(HELDOUT_TASKS)].notna().sum(axis=1)  / n_heldout if n_heldout else float("nan")

    tasks_found = [t for t in ALL_TASKS if f"probe_acc_{safe_col(t)}" in df.columns]
    tasks_missing = [t for t in ALL_TASKS if f"probe_acc_{safe_col(t)}" not in df.columns]
    if tasks_missing:
        print(f"  Tasks missing from probe data: {tasks_missing}")
    print(f"  Tasks found in probe data: {len(tasks_found)}/{len(ALL_TASKS)}")
    print(f"  Probe rows loaded: {len(df)}")
    return df


def load_cohort_data(cohort_path: Path) -> pd.DataFrame:
    df = pd.read_csv(cohort_path)
    cohort_series = (
        df.groupby("run_id")["cohort"]
        .apply(lambda x: "/".join(sorted(set(x))))
        .reset_index()
    )
    cohort_series.columns = ["run_id", "cohort"]
    print(f"Cohort assignments: {len(cohort_series)} unique run_ids")
    print(f"  Distribution: {dict(df['cohort'].value_counts())}")
    return cohort_series


TASK_METADATA = [
    # ── Training — Reach ──────────────────────────────────────────────────────
    # task, family, n_mods, is_heldout, dly, anti, ctx, rt, multi, cat, nms, ctx_t, anti_t, double_t, heldout_group
    ("poli.go",           "reach",    0, False, False, False, False, False, False, False, False, False, False, False, None),
    ("poli.rtgo",         "reach",    1, False, False, False, False, True,  False, False, False, False, False, False, None),
    ("poli.dlygo",        "reach",    1, False, True,  False, False, False, False, False, False, False, False, False, None),
    ("poli.antigo",       "reach",    1, False, False, True,  False, False, False, False, False, False, False, False, None),
    ("poli.ctxgo",        "reach",    1, False, False, False, True,  False, False, False, False, False, False, False, None),
    ("poli.dlyantigo",    "reach",    2, False, True,  True,  False, False, False, False, False, False, False, False, None),
    ("poli.dlyctxgo",     "reach",    2, False, True,  False, True,  False, False, False, False, False, False, False, None),
    ("poli.antictxgo",    "reach",    2, False, False, True,  True,  False, False, False, False, False, False, False, None),
    ("poli.dlyantictxgo", "reach",    3, False, True,  True,  True,  False, False, False, False, False, False, False, None),
    ("poli.rtantictxgo",  "reach",    3, False, False, True,  True,  True,  False, False, False, False, False, False, None),
    # ── Training — Decision ───────────────────────────────────────────────────
    ("poli.dm1",          "decision", 0, False, False, False, False, False, False, False, False, False, False, False, None),
    ("poli.dm2",          "decision", 0, False, False, False, False, False, False, False, False, False, False, False, None),
    ("poli.dlydm1",       "decision", 1, False, True,  False, False, False, False, False, False, False, False, False, None),
    ("poli.multidm",      "decision", 1, False, False, False, False, False, True,  False, False, False, False, False, None),
    ("poli.antidm1",      "decision", 1, False, False, True,  False, False, False, False, False, False, False, False, None),
    ("poli.antidlydm1",   "decision", 2, False, True,  True,  False, False, False, False, False, False, False, False, None),
    # ── Training — Match ──────────────────────────────────────────────────────
    ("poli.dlyms",        "match",    1, False, True,  False, False, False, False, False, False, False, False, False, None),
    ("poli.dlynms",       "match",    2, False, True,  False, False, False, False, False, True,  False, False, False, None),
    ("poli.catdlyms",     "match",    2, False, True,  False, False, False, False, True,  False, False, False, False, None),
    ("poli.catdlynms",    "match",    3, False, True,  False, False, False, False, True,  True,  False, False, False, None),
    # ── Held-out — ctx → Decision ─────────────────────────────────────────────
    ("poli.ctxdm1",          "decision", 1, True, False, False, True,  False, False, False, False, True,  False, False, "ctx_decision"),
    ("poli.ctxdm2",          "decision", 1, True, False, False, True,  False, False, False, False, True,  False, False, "ctx_decision"),
    ("poli.ctxdlydm1",       "decision", 2, True, True,  False, True,  False, False, False, False, True,  False, False, "ctx_decision"),
    ("poli.ctxdlydm2",       "decision", 2, True, True,  False, True,  False, False, False, False, True,  False, False, "ctx_decision"),
    ("poli.antictxdm1",      "decision", 2, True, False, True,  True,  False, False, False, False, True,  False, False, "ctx_decision"),
    ("poli.antictxdm2",      "decision", 2, True, False, True,  True,  False, False, False, False, True,  False, False, "ctx_decision"),
    ("poli.antictxdlydm1",   "decision", 3, True, True,  True,  True,  False, False, False, False, True,  False, False, "ctx_decision"),
    ("poli.antictxdlydm2",   "decision", 3, True, True,  True,  True,  False, False, False, False, True,  False, False, "ctx_decision"),
    # ── Held-out — anti → Match ───────────────────────────────────────────────
    ("poli.antidlyms",       "match",    2, True, True,  True,  False, False, False, False, False, False, True,  False, "anti_match"),
    ("poli.antidlynms",      "match",    3, True, True,  True,  False, False, False, False, True,  False, True,  False, "anti_match"),
    # ── Held-out — ctx + anti → Match (double transport) ─────────────────────
    # Both ctx AND anti are new to Match → ctx_transported=True AND anti_transported=True
    ("poli.antictxdlyms",    "match",    3, True, True,  True,  True,  False, False, False, False, True,  True,  True,  "double_transport"),
    ("poli.antictxcatdlyms", "match",    4, True, True,  True,  True,  False, False, True,  False, True,  True,  True,  "double_transport"),
]

TASK_META_COLS = [
    "task", "family", "n_mods", "is_heldout",
    "has_dly", "has_anti", "has_ctx", "has_rt", "has_multi", "has_cat", "has_nms",
    "ctx_transported", "anti_transported", "double_transported", "heldout_group",
]


def write_task_metadata(out_dir: Path):
    df = pd.DataFrame(TASK_METADATA, columns=TASK_META_COLS)
    path = out_dir / "task_metadata.csv"
    df.to_csv(path, index=False)
    print(f"Saved task metadata ({len(df)} tasks) -> {path.name}")
    return df


def main():
    # ── Load base ─────────────────────────────────────────────────────────────
    print(f"Loading {CSV_PATH.name} ...")
    df_grand = pd.read_csv(CSV_PATH)
    print(f"  {len(df_grand)} rows, {df_grand.shape[1]} columns")

    # ── Cohort assignments ────────────────────────────────────────────────────
    print(f"\nLoading cohort map ...")
    df_cohort = load_cohort_data(COHORT_PATH)

    # ── Probe data ────────────────────────────────────────────────────────────
    print(f"\nScanning {SWEEP_DIR} for probe results ...")
    df_probe = load_probe_data(SWEEP_DIR)

    if df_probe.empty:
        print("No probe data found — aborting.")
        return

    # ── Merge (inner join on probe: only keep probed models) ──────────────────
    print("\nMerging ...")
    df = df_grand.merge(df_probe, on="run_id", how="inner")
    print(f"  After inner join with probe data: {len(df)} rows")

    df = df.merge(df_cohort, on="run_id", how="left")
    print(f"  After left join with cohort map:  {len(df)} rows")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nFinal dataset: {len(df)} rows × {df.shape[1]} columns")
    print(f"  Cohort distribution:\n{df['cohort'].value_counts(dropna=False).to_string()}")
    print(f"  probe_mean_acc         : {df['probe_mean_acc'].mean():.3f} ± {df['probe_mean_acc'].std():.3f}")
    print(f"  probe_mean_acc_training: {df['probe_mean_acc_training'].mean():.3f} ± {df['probe_mean_acc_training'].std():.3f}")
    print(f"  probe_mean_acc_heldout : {df['probe_mean_acc_heldout'].mean():.3f} ± {df['probe_mean_acc_heldout'].std():.3f}")

    # ── Task metadata ─────────────────────────────────────────────────────────
    print("\nWriting task metadata ...")
    write_task_metadata(DATA_DIR)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

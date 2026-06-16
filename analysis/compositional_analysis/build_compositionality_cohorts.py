#!/usr/bin/env python3
"""
Build the four compositionality cohorts from grand_unified_metrics_v2.csv.

Cohorts:
  H-{K}  : Top K most human-like models (lowest DISTANCE_COL, fraction_solved >= MIN_SOLVED)
  C-temp : Matched controls -- same params but temp = 1.0
  C-reg  : Matched controls -- same params but reg_value = minimum available for that reg_type
             (L1 -> 1e-06, Distance -> 1e-05)
  C-both : Matched controls -- same params but temp = 1.0 AND reg_value = minimum

Outputs (under analysis/compositional_analysis/Data/cohorts/):
  - Summary table printed to stdout
  - h{K}_ids.txt, c_temp_ids.txt, c_reg_ids.txt, c_both_ids.txt
    (one numeric run ID per line -- ready to paste into --target-runs)
  - all_ids.txt  (all four cohorts combined)
  - cohort_map.csv  (full mapping with parameters)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]   # compositional_analysis -> analysis -> project root
CSV_PATH   = ROOT / "analysis" / "grand_unified_metrics_v2.csv"
HCP_CSV    = ROOT / "analysis" / "Network_analysis" / "Results" / "human_topological_metrics.csv"
OUT_DIR    = ROOT / "analysis" / "compositional_analysis" / "Data" / "cohorts"
K                = 100
MIN_SOLVED       = 0.3    # broad pre-filter: pool to select top-K from (matches topology pool)
MIN_SOLVED_STRICT = 0.9   # second cut: keep only top-K models that also pass this threshold

# "Most human-like" = smallest Mahalanobis distance to the HCP-YA average over the
# THREE topological metrics below — identical fingerprint to
# pub_topology_consolidated.R / pub_synaptic_consolidated.R. (Replaces the legacy
# norm_dist_network z-Euclidean distance over the old metric implementations.)
METRIC_COLS      = ["modularity_leiden", "efficiency", "rich_club"]
DISTANCE_COL     = "mahal_dist"

CONTROL_TEMP = 1.0
# Minimum reg_value available per reg_type (Distance has no 1e-06 level)
MIN_REG = {"L1": 1e-6, "Distance": 1e-5}

# Parameters that define the "identity" of a model (matched across cohorts)
MATCH_COLS = ["beta", "alpha", "reg_type", "travel", "eps"]


def extract_numeric_id(run_id: str):
    m = re.match(r"^run_(\d+)_", run_id)
    return int(m.group(1)) if m else None


def mahalanobis_to_human(df: pd.DataFrame) -> pd.Series:
    """Mahalanobis distance of each model's (METRIC_COLS) fingerprint to the
    HCP-YA individual average, using the inter-individual covariance — the same
    fingerprint used by pub_topology_consolidated.R. NaN where metrics missing."""
    hcp = pd.read_csv(HCP_CSV)
    hcp = hcp[(~hcp["full_id"].astype(str).str.contains("Consensus")) &
              (hcp["dataset"] == "HCPya")]
    mu   = hcp[METRIC_COLS].mean().to_numpy()
    Sinv = np.linalg.inv(np.cov(hcp[METRIC_COLS].to_numpy(), rowvar=False))
    X = df[METRIC_COLS].to_numpy(dtype=float) - mu
    d = np.sqrt(np.einsum("ij,jk,ik->i", X, Sinv, X))
    d[~np.isfinite(df[METRIC_COLS].to_numpy(dtype=float)).all(axis=1)] = np.nan
    return pd.Series(d, index=df.index)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df = df[df["fraction_solved"] >= MIN_SOLVED].copy()
    df["numeric_id"] = df["run_id"].apply(extract_numeric_id)
    df[DISTANCE_COL] = mahalanobis_to_human(df)   # Mahalanobis over METRIC_COLS
    df = df.dropna(subset=[DISTANCE_COL, "numeric_id"])
    df["numeric_id"] = df["numeric_id"].astype(int)

    print(f"Loaded {len(df)} functional models (fraction_solved >= {MIN_SOLVED})")

    h_label = f"H-{K}"

    # ── H-{K} ────────────────────────────────────────────────────────────────
    # 1. Take top-K by DISTANCE_COL from the broad pool (fraction_solved >= MIN_SOLVED)
    # 2. Keep only those also passing the strict performance threshold (fraction_solved >= MIN_SOLVED_STRICT)
    # 3. Apply the same structural filter as chisquare_analysis.R:
    #    temp < 0.01  (deterministic-enough switching)
    #    reg_value > min for reg_type  (L1: 1e-6, Distance: 1e-5)
    topK = df.nsmallest(K, DISTANCE_COL).copy()
    print(f"\nTop-{K} by {DISTANCE_COL} (from pool fraction_solved >= {MIN_SOLVED}): {len(topK)}")
    topK_strict = topK[topK["fraction_solved"] >= MIN_SOLVED_STRICT].copy()
    print(f"After strict performance filter (fraction_solved >= {MIN_SOLVED_STRICT}): {len(topK_strict)}")
    topK_strict["min_reg"] = topK_strict["reg_type"].map(MIN_REG)
    h50 = topK_strict[(topK_strict["temp"] < 0.01) & (topK_strict["reg_value"] > topK_strict["min_reg"])].copy()
    h50 = h50.drop(columns=["min_reg"])
    h50["cohort"] = h_label
    print(f"{h_label} after structural filter (temp<0.01 & reg>min): {len(h50)}")
    print(f"  {DISTANCE_COL} range: [{h50[DISTANCE_COL].min():.4f}, {h50[DISTANCE_COL].max():.4f}]")
    print(f"  temp    : {dict(h50['temp'].value_counts())}")
    print(f"  reg_type: {dict(h50['reg_type'].value_counts())}")
    print(f"  reg_val : {dict(h50['reg_value'].value_counts())}")
    print(f"  eps     : {dict(h50['eps'].value_counts())}")

    # ── Build control lookups ─────────────────────────────────────────────────
    df_idx    = df.set_index(MATCH_COLS + ["temp", "reg_value"])["numeric_id"]
    df_by_nid = df.set_index("numeric_id")   # used to retrieve control model's own row

    def find_control(row, new_temp=None, new_reg=None):
        """Return numeric_id of the matched control, or None if not in sweep."""
        t  = new_temp if new_temp is not None else row["temp"]
        rv = new_reg  if new_reg  is not None else row["reg_value"]
        key = tuple(row[c] for c in MATCH_COLS) + (t, rv)
        try:
            val = df_idx.loc[key]
            if isinstance(val, pd.Series):
                return int(val.iloc[0])
            return int(val)
        except KeyError:
            return None

    # ── C-temp ────────────────────────────────────────────────────────────────
    c_temp_ids, c_temp_rows = [], []
    for _, row in h50.iterrows():
        if row["temp"] == CONTROL_TEMP:
            print(f"  SKIP C-temp: {row['run_id']} already has temp={CONTROL_TEMP}")
            continue
        nid = find_control(row, new_temp=CONTROL_TEMP)
        if nid is not None:
            c_temp_ids.append(nid)
            ctrl = df_by_nid.loc[nid].to_dict()
            c_temp_rows.append({**ctrl, "numeric_id": nid, "cohort": "C-temp"})
        else:
            print(f"  MISSING C-temp match for {row['run_id']}")

    # ── C-reg ─────────────────────────────────────────────────────────────────
    # Use minimum available reg_value per reg_type (L1->1e-6, Distance->1e-5)
    c_reg_ids, c_reg_rows = [], []
    for _, row in h50.iterrows():
        target_rv = MIN_REG[row["reg_type"]]
        if row["reg_value"] == target_rv:
            print(f"  SKIP C-reg: {row['run_id']} already at minimum reg_value ({target_rv})")
            continue
        nid = find_control(row, new_reg=target_rv)
        if nid is not None:
            c_reg_ids.append(nid)
            ctrl = df_by_nid.loc[nid].to_dict()
            c_reg_rows.append({**ctrl, "numeric_id": nid, "cohort": "C-reg"})
        else:
            print(f"  MISSING C-reg match for {row['run_id']}")

    # ── C-both ────────────────────────────────────────────────────────────────
    c_both_ids, c_both_rows = [], []
    for _, row in h50.iterrows():
        target_rv = MIN_REG[row["reg_type"]]
        if row["temp"] == CONTROL_TEMP and row["reg_value"] == target_rv:
            print(f"  SKIP C-both: {row['run_id']} already matches C-both")
            continue
        nid = find_control(row, new_temp=CONTROL_TEMP, new_reg=target_rv)
        if nid is not None:
            c_both_ids.append(nid)
            ctrl = df_by_nid.loc[nid].to_dict()
            c_both_rows.append({**ctrl, "numeric_id": nid, "cohort": "C-both"})
        else:
            print(f"  MISSING C-both match for {row['run_id']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'Cohort':<10}  {'N found':>8}  {'N missing':>9}  Notes")
    print(f"{'-'*55}")
    print(f"{h_label:<10}  {len(h50):>8}  {'0':>9}")
    print(f"{'C-temp':<10}  {len(c_temp_ids):>8}  {K - len(c_temp_ids):>9}  temp -> 1.0")
    print(f"{'C-reg':<10}  {len(c_reg_ids):>8}  {K - len(c_reg_ids):>9}  reg_value -> min (L1:1e-6, Dist:1e-5)")
    print(f"{'C-both':<10}  {len(c_both_ids):>8}  {K - len(c_both_ids):>9}  both changes")
    print(f"{'='*55}")

    # ── Write ID files ────────────────────────────────────────────────────────
    h_key = f"h{K}"
    cohort_ids = {
        h_key:    list(h50["numeric_id"].astype(int)),
        "c_temp": c_temp_ids,
        "c_reg":  c_reg_ids,
        "c_both": c_both_ids,
    }

    # Deduplicate each cohort (multiple H-{K} models can map to the same control)
    cohort_ids = {name: sorted(set(ids)) for name, ids in cohort_ids.items()}
    all_ids = sorted(set(
        cohort_ids[h_key] + cohort_ids["c_temp"] + cohort_ids["c_reg"] + cohort_ids["c_both"]
    ))

    for name, ids in cohort_ids.items():
        out = OUT_DIR / f"{name}_ids.txt"
        out.write_text("\n".join(str(i) for i in ids))
        print(f"Saved {len(ids)} unique IDs -> {out.name}")

    (OUT_DIR / "all_ids.txt").write_text("\n".join(str(i) for i in all_ids))
    print(f"Saved {len(all_ids)} unique IDs -> all_ids.txt")

    # ── Print ready-to-use commands ───────────────────────────────────────────
    sweep = "/imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6"
    all_id_str = " ".join(str(i) for i in all_ids)
    print("\n" + "="*55)
    print("CLUSTER COMMAND (run from project root on login-k01)")
    print("="*55)
    print(f"\n# All {len(all_ids)} unique IDs across all 4 cohorts ({h_label}, C-temp, C-reg, C-both)")
    print("python analysis/compare_experiments_forage.py \\")
    print(f"    --sweep_dirs {sweep} \\")
    print("    --mode targeted \\")
    print(f"    --target-runs {all_id_str} \\")
    print("    --compute-gen --gen-set full --probe-steps 5000 \\")
    print("    --execution-mode cluster")

    # ── Save mapping CSV ──────────────────────────────────────────────────────
    keep_cols = ["run_id", "numeric_id", "beta", "alpha", "temp",
                 "reg_type", "reg_value", "travel", "eps", DISTANCE_COL, "cohort"]
    frames = [h50[keep_cols].copy()]
    for rows in [c_temp_rows, c_reg_rows, c_both_rows]:
        if rows:
            frames.append(pd.DataFrame(rows)[keep_cols])
    all_rows = pd.concat(frames, ignore_index=True)
    map_csv = OUT_DIR / "cohort_map.csv"
    all_rows.to_csv(map_csv, index=False)
    print(f"\nFull cohort map saved -> {map_csv}")


if __name__ == "__main__":
    main()

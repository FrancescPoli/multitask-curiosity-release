"""
Compare L1 vs Distance branches on input / readout / recurrent weight norms.

Implements the diagnostic in future_steps.md section 4.1.
Inputs:
  - analysis/Regularization_scope/io_weight_norms.csv
  - analysis/Regularization_scope/io_weight_norms_init.csv
  - analysis/grand_unified_metrics_v2.csv
Outputs:
  - analysis/Regularization_scope/figures/*.png
  - prints summary statistics + paired Wilcoxon results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NORMS = ["w_in_l1_sum", "w_rec_l1_sum", "w_out_l1_sum"]
LABELS = {
    "w_in_l1_sum":  "W_in (unregularized in Distance branch)",
    "w_rec_l1_sum": "W_rec (penalized in both)",
    "w_out_l1_sum": "W_out (unregularized in Distance branch)",
}
MATCH_KEYS = ["beta", "alpha", "travel", "temp", "eps"]


def load() -> tuple[pd.DataFrame, dict]:
    norms = pd.read_csv(HERE / "io_weight_norms.csv")
    init = pd.read_csv(HERE / "io_weight_norms_init.csv").iloc[0].to_dict()
    grand = pd.read_csv(HERE.parent / "grand_unified_metrics_v2.csv")
    df = norms.merge(grand, on="run_id", how="inner")
    print(f"Merged: {len(df)} rows (norms={len(norms)}, grand={len(grand)})")
    return df, init


def summary_by_branch(df: pd.DataFrame, init: dict) -> None:
    print("\n=== Per-branch summary (median ; ratio to init) ===")
    rows = []
    for col in NORMS:
        ref = init[col]
        for branch, sub in df.groupby("reg_type"):
            m = sub[col].median()
            rows.append({
                "branch": branch,
                "layer": col,
                "n": len(sub),
                "median": round(m, 1),
                "ratio_to_init": round(m / ref, 2),
                "init": round(ref, 1),
            })
    print(pd.DataFrame(rows).to_string(index=False))


def plot_inflation_violin(df: pd.DataFrame, init: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, col in zip(axes, NORMS):
        ref = init[col]
        data = [df.loc[df["reg_type"] == b, col] / ref for b in ("L1", "Distance")]
        parts = ax.violinplot(data, showmedians=True)
        for pc, c in zip(parts["bodies"], ("tab:blue", "tab:orange")):
            pc.set_facecolor(c); pc.set_alpha(0.6)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["L1", "Distance"])
        ax.axhline(1.0, color="k", lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.set_title(LABELS[col]); ax.set_ylabel("final / init")
    fig.suptitle("Layer-norm inflation relative to initialization, by branch")
    fig.tight_layout()
    out = FIG_DIR / "inflation_by_branch.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


def matched_contrast(df: pd.DataFrame) -> None:
    """Paired Wilcoxon L1 vs Distance on matched foraging hyperparams.
    Within each (beta, alpha, travel, temp, eps) cell, average over the
    multiple reg_value choices in each branch, then pair across branches."""
    print("\n=== Matched-config L1 vs Distance (paired Wilcoxon over foraging cells) ===")
    agg = (
        df.groupby(MATCH_KEYS + ["reg_type"])[NORMS]
        .median()
        .reset_index()
    )
    wide = agg.pivot_table(index=MATCH_KEYS, columns="reg_type", values=NORMS)
    rows = []
    for col in NORMS:
        try:
            a = wide[(col, "L1")].dropna()
            b = wide[(col, "Distance")].dropna()
            common = a.index.intersection(b.index)
            a = a.loc[common]; b = b.loc[common]
            if len(common) < 5:
                rows.append({"layer": col, "n_cells": len(common), "note": "too few cells"}); continue
            stat, p = wilcoxon(a.values, b.values)
            rows.append({
                "layer": col,
                "n_cells": len(common),
                "median_L1": round(a.median(), 1),
                "median_Dist": round(b.median(), 1),
                "Dist - L1 (median delta)": round((b - a).median(), 1),
                "wilcoxon_p": f"{p:.3g}",
            })
        except Exception as e:
            rows.append({"layer": col, "note": f"failed: {e}"})
    print(pd.DataFrame(rows).to_string(index=False))


def plot_matched_scatter(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(MATCH_KEYS + ["reg_type"])[NORMS]
        .median()
        .reset_index()
    )
    wide = agg.pivot_table(index=MATCH_KEYS, columns="reg_type", values=NORMS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, NORMS):
        a = wide[(col, "L1")]; b = wide[(col, "Distance")]
        common = a.dropna().index.intersection(b.dropna().index)
        ax.scatter(a.loc[common], b.loc[common], alpha=0.5, s=15)
        lo = min(a.loc[common].min(), b.loc[common].min())
        hi = max(a.loc[common].max(), b.loc[common].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        ax.set_xlabel("L1 branch (median over reg_value)")
        ax.set_ylabel("Distance branch")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(LABELS[col])
    fig.suptitle("Matched-config L1 vs Distance — points above diagonal = Distance larger")
    fig.tight_layout()
    out = FIG_DIR / "matched_scatter.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


def plot_io_vs_performance(df: pd.DataFrame) -> None:
    """For each branch x I/O layer, scatter fraction_solved vs final L1 sum.
    Shows the sign flip: in Distance, inflated I/O is associated with failure."""
    layers = ["w_in_l1_sum", "w_out_l1_sum"]
    branches = ["L1", "Distance"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    for r, col in enumerate(layers):
        for c, branch in enumerate(branches):
            ax = axes[r, c]
            sub = df[df["reg_type"] == branch].dropna(subset=[col, "fraction_solved", "temp"])
            sc = ax.scatter(sub[col], sub["fraction_solved"],
                            c=np.log10(sub["temp"]), cmap="viridis",
                            s=10, alpha=0.5)
            r_s, p_s = spearmanr(sub[col], sub["fraction_solved"])
            ax.set_xscale("log")
            ax.set_title(f"{branch} branch — {col}\nSpearman rho = {r_s:.2f} (p = {p_s:.1e}, n = {len(sub)})")
            ax.set_xlabel(f"{col}  (final L1 sum)")
            if c == 0:
                ax.set_ylabel("fraction_solved")
            ax.axhline(0.9, color="grey", lw=0.5, ls=":")
        cbar = fig.colorbar(sc, ax=axes[r, :].tolist(), shrink=0.85, pad=0.02)
        cbar.set_label("log10(forage_temperature)")
    fig.suptitle("I/O weight inflation vs task performance, by branch")
    out = FIG_DIR / "io_norms_vs_fraction_solved.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


def plot_io_vs_performance_binned(df: pd.DataFrame) -> None:
    """Same finding but with binned medians — cleaner trend lines."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["w_in_l1_sum", "w_out_l1_sum"]):
        for branch, color in [("L1", "tab:blue"), ("Distance", "tab:orange")]:
            sub = df[df["reg_type"] == branch].dropna(subset=[col, "fraction_solved"])
            sub = sub.copy()
            sub["bin"] = pd.qcut(sub[col], q=15, duplicates="drop")
            grp = sub.groupby("bin", observed=True).agg(
                x=(col, "median"),
                y=("fraction_solved", "median"),
                lo=("fraction_solved", lambda s: s.quantile(0.25)),
                hi=("fraction_solved", lambda s: s.quantile(0.75)),
            ).reset_index(drop=True)
            ax.plot(grp["x"], grp["y"], "o-", color=color, label=branch)
            ax.fill_between(grp["x"], grp["lo"], grp["hi"], color=color, alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel(f"{col}  (final L1 sum, log)")
        ax.set_ylabel("fraction_solved (median, IQR shaded)")
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Branches occupy non-overlapping I/O-norm regimes; Distance's high tail fails for W_in")
    fig.tight_layout()
    out = FIG_DIR / "io_norms_vs_fraction_solved_binned.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


def correlations_with_outcomes(df: pd.DataFrame) -> None:
    print("\n=== Spearman correlations within each branch ===")
    outcomes = ["fraction_solved", "mean_accuracy", "dist_synaptic"]
    rows = []
    for branch, sub in df.groupby("reg_type"):
        for col in NORMS:
            for out in outcomes:
                if out not in sub: continue
                m = sub[[col, out]].dropna()
                if len(m) < 20: continue
                r, p = spearmanr(m[col], m[out])
                rows.append({
                    "branch": branch, "layer": col, "outcome": out,
                    "rho": round(r, 3), "p": f"{p:.2g}", "n": len(m),
                })
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_solved", type=float, default=None,
                        help="Filter to runs with fraction_solved >= this (None = no filter)")
    args = parser.parse_args()

    df, init = load()
    if args.min_solved is not None:
        before = len(df)
        df = df[df["fraction_solved"] >= args.min_solved].copy()
        print(f"Filtered fraction_solved >= {args.min_solved}: {len(df)}/{before} rows")

    print("\nInit reference:", {k: round(v, 1) for k, v in init.items() if k.startswith("w_")})
    summary_by_branch(df, init)
    plot_inflation_violin(df, init)
    matched_contrast(df)
    plot_matched_scatter(df)
    plot_io_vs_performance(df)
    plot_io_vs_performance_binned(df)
    correlations_with_outcomes(df)
    print(f"\nFigures saved under: {FIG_DIR}")


if __name__ == "__main__":
    main()

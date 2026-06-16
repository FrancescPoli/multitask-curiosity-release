"""
Publication-quality connectivity matrices for Figure 3.

Produces individual weighted adjacency-matrix heatmaps (PNG only) in
  analysis/Final_analyses_and_plots/Figures/matrix_plots/
with:
  - top N_MODELS H-100 runs (lowest norm_dist_network)
  - N_HUMANS randomly sampled HCPya subjects (Schaefer100, age 26-35)

Human side uses the Schaefer100 harmonized_connectomes pipeline
(analysis/Network_analysis/extract_schaefer_metrics.py), which is what
norm_dist_network in compositional_dataset.csv is actually computed from.

Matrices are cached as .npy in conn_matrices_cache/ so re-runs are instant.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Tweakables ───────────────────────────────────────────────────────────────
FIG_W = 2.8          # width  (inches) per heatmap
FIG_H = 2.8          # height (inches) per heatmap
DPI   = 400

CMAP_NAME  = "magma_r"   # weighted heatmap colormap
VMIN_PCTL  = 0           # lower percentile for colour scale
VMAX_PCTL  = 99          # clip top percentile (suppresses outliers)

# Show the matrix as the topology metrics actually "see" it: prune to a target
# density (keep strongest edges) and mean-normalize, exactly like
# topology_metrics._prepare. Set PRUNE_DENSITY = None to show the full matrix.
PRUNE_DENSITY  = 0.135   # matches TARGET_DENSITY in topology_metrics.py
MEAN_NORMALIZE = True    # divide surviving weights by their mean (metrics do this)

# The three metrics defining "human-likeness" (match pub_topology_consolidated.R).
# Models are ranked by Mahalanobis distance to the HCPya consensus over these,
# recomputed from the upgraded topology CSVs (not the old norm_dist_network).
METRIC_COLS = ["modularity_leiden", "efficiency", "rich_club"]

N_MODELS = 5             # top-N H-100 runs by Mahalanobis distance to humans
N_HUMANS = 10            # random HCPya subjects (26-35y)
HUMAN_SEED = 42          # deterministic subject sampling

COHORT         = "H-100"
COMP_CSV       = "analysis/compositional_analysis/Data/compositional_dataset.csv"   # cohort membership
RNN_CSV        = "analysis/Network_analysis/Results/rnn_topological_metrics.csv"  # metric values
HCP_CSV        = "analysis/Network_analysis/Results/human_topological_metrics.csv"  # HCPya reference
SWEEP_DIR      = Path("Z:/fp02/logs/sweep/forage_v5.1/forage_v6")

# Schaefer100 pipeline (matches extract_schaefer_metrics.py, the canonical human side).
SCHAEFER_MAT   = Path("analysis/Network_analysis/Data/Atlases/Schaefer/harmonized_connectomes.mat")
SCHAEFER_AGE   = Path("analysis/Network_analysis/Data/Atlases/Schaefer/age.mat")

OUT_DIR    = Path("analysis/Final_analyses_and_plots/Figures")
PLOTS_DIR  = OUT_DIR / "matrix_plots"
CACHE_DIR  = Path("analysis/Final_analyses_and_plots") / "conn_matrices_cache"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Model W_rec ──────────────────────────────────────────────────────────────
def pick_top_model_runs(n: int) -> pd.DataFrame:
    """Top-n cohort runs by Mahalanobis distance to the HCPya consensus,
    computed over METRIC_COLS from the upgraded topology CSVs."""
    # cohort membership (run_ids only) x upgraded metric values
    comp = pd.read_csv(COMP_CSV, low_memory=False)
    runs = comp.loc[comp["cohort"] == COHORT, ["run_id"]].drop_duplicates()
    rnn = pd.read_csv(RNN_CSV)
    miss = [c for c in METRIC_COLS if c not in rnn.columns]
    if miss:
        raise RuntimeError(f"{RNN_CSV} missing METRIC_COLS: {miss}")
    m = (runs.merge(rnn[["run_id", *METRIC_COLS]], on="run_id", how="inner")
             .dropna(subset=METRIC_COLS))
    if m.empty:
        raise RuntimeError(f"No {COHORT} runs with {METRIC_COLS} after merge")

    # HCPya individual reference (mean + covariance)
    hcp = pd.read_csv(HCP_CSV)
    hcp = hcp[(~hcp["full_id"].astype(str).str.contains("Consensus")) &
              (hcp["dataset"] == "HCPya")]
    mu = hcp[METRIC_COLS].mean().to_numpy()
    Sinv = np.linalg.inv(np.cov(hcp[METRIC_COLS].to_numpy(), rowvar=False))
    X = m[METRIC_COLS].to_numpy() - mu
    m["mahal_dist"] = np.sqrt(np.einsum("ij,jk,ik->i", X, Sinv, X))

    sub = m.sort_values("mahal_dist").head(n).reset_index(drop=True)
    print(f"Top {len(sub)} {COHORT} runs by Mahalanobis distance "
          f"({', '.join(METRIC_COLS)}):")
    for i, r in sub.iterrows():
        print(f"  [{i+1}] {r['run_id']}  ({r['mahal_dist']:.4f})")
    return sub[["run_id", "mahal_dist"]]


def load_model_conn(run_id: str) -> np.ndarray:
    cache = CACHE_DIR / f"model_{run_id}.npy"
    if cache.exists():
        return np.load(cache)

    import torch
    from analysis.plotting.utils import (
        load_meta, build_model_from_meta, load_state_into_model, get_Wrec,
    )

    run_dir = SWEEP_DIR / run_id
    meta  = load_meta(run_dir)
    model = build_model_from_meta(meta, device="cpu")
    ckpt  = torch.load(run_dir / "model_last.pt", map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    load_state_into_model(model, state)

    W = get_Wrec(model).detach().abs().cpu().numpy()
    np.fill_diagonal(W, 0.0)
    np.save(cache, W)
    print(f"  extracted W_rec {W.shape} -> {cache.name}")
    return W


# ── HCPya individual subjects (Schaefer100 harmonized connectomes) ───────────
# Matches extract_schaefer_metrics.py: indices 636-1700 = HCPya; filter age 26-35.
def load_human_samples(n: int, seed: int) -> dict[str, np.ndarray]:
    cache = CACHE_DIR / f"schaefer100_hcpya_random{n}_seed{seed}.npz"
    if cache.exists():
        data = np.load(cache)
        mats = {k: data[k] for k in data.files}
        print(f"  load cached: {cache.name} ({len(mats)} subjects)")
        return mats

    import h5py

    with h5py.File(SCHAEFER_AGE, "r") as f:
        ages = f["age"][()].flatten()
    with h5py.File(SCHAEFER_MAT, "r") as f:
        C = f["harmonized_connectomes"][()]   # (100, 100, 2419)

    is_hcpya = np.zeros(len(ages), dtype=bool)
    is_hcpya[636:1701] = True
    mask = is_hcpya & (ages >= 26) & (ages <= 35)
    all_idx = np.where(mask)[0]
    print(f"  Schaefer100 HCPya 26-35y pool: {len(all_idx)}")

    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(all_idx, size=n, replace=False))
    print(f"  sampled idx: {pick.tolist()}")
    sub_mats = C[:, :, pick]

    mats = {}
    for i, idx in enumerate(pick):
        M = sub_mats[:, :, i].astype(float)
        np.fill_diagonal(M, 0.0)
        mats[f"subj{int(idx):05d}"] = M
    np.savez(cache, **mats)
    print(f"  cached {len(mats)} subject matrices -> {cache.name}")
    return mats


# ── Plot ─────────────────────────────────────────────────────────────────────
def prune_to_density(M: np.ndarray, density: float) -> np.ndarray:
    """Keep the strongest edges to reach `density` (matches
    topology_metrics.prune_network: if already sparser, leave as-is)."""
    M = M.copy()
    np.fill_diagonal(M, 0.0)
    iu = np.triu_indices(M.shape[0], 1)
    w = M[iu]
    n_possible = w.size
    if n_possible == 0:
        return M
    cur = np.count_nonzero(w) / n_possible
    if 0 < cur < density:
        return M
    n_keep = int(n_possible * density)
    sorted_w = np.sort(w)[::-1]
    thr = sorted_w[n_keep] if n_keep < sorted_w.size else 0.0
    M[M < thr] = 0.0
    return M


def _short_num(x, _pos=None):
    """Format ticks: 12345 -> '12k', 1.2e6 -> '1.2M'. Small values kept as-is."""
    ax = abs(x)
    if ax >= 1e6:   return f"{x/1e6:g}M"
    if ax >= 1e3:   return f"{x/1e3:g}k"
    if ax >= 1:     return f"{x:g}"
    return f"{x:.2g}"


def plot_matrix(W: np.ndarray, out_stem: Path, title: str | None = None) -> None:
    M = W.copy()
    M = (M + M.T) / 2.0          # symmetrise: i->j and j->i made equal
    np.fill_diagonal(M, 0.0)
    if PRUNE_DENSITY is not None:    # match what the topology metrics see
        M = prune_to_density(M, PRUNE_DENSITY)
        if MEAN_NORMALIZE:
            active = M[M > 0]
            if active.size:
                M = M / active.mean()
    pos = M[M > 0]
    if pos.size:
        vmin = np.percentile(pos, VMIN_PCTL)
        vmax = np.percentile(pos, VMAX_PCTL)
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    im = ax.imshow(M, cmap=CMAP_NAME, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("black")
    if title is not None:
        ax.set_title(title, fontsize=9, pad=4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.4)
    cbar.outline.set_linewidth(0.5)
    cbar.formatter = plt.FuncFormatter(_short_num)
    cbar.update_ticks()

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{out_stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_stem.name}.png")


def main():
    print(f">>> Model W_rec (top {N_MODELS} by Mahalanobis distance)")
    # Pull a buffer and skip runs whose checkpoint is missing/unreadable,
    # substituting the next-closest, until N_MODELS are saved.
    pool = pick_top_model_runs(N_MODELS * 4)
    # clear previous model_*.png so stale ranks/run_ids don't accumulate
    for old in PLOTS_DIR.glob("model_*.png"):
        old.unlink()
    saved = 0
    for _, row in pool.iterrows():
        if saved >= N_MODELS:
            break
        try:
            W = load_model_conn(row["run_id"])
        except (FileNotFoundError, OSError) as e:
            print(f"  skip {row['run_id']}: no loadable checkpoint ({type(e).__name__})")
            continue
        saved += 1
        plot_matrix(
            W,
            PLOTS_DIR / f"model_{saved:02d}_{row['run_id']}",
            title=f"Model #{saved}  ({W.shape[0]} units, dist={row['mahal_dist']:.3f})",
        )
    if saved < N_MODELS:
        print(f"  WARNING: only {saved}/{N_MODELS} models had loadable checkpoints")

    print(">>> HCPya random subjects")
    humans = load_human_samples(N_HUMANS, HUMAN_SEED)
    for i, (sid, W) in enumerate(humans.items()):
        plot_matrix(
            W,
            PLOTS_DIR / f"human_{i+1:02d}_{sid}",
            title=f"HCPya {sid}  ({W.shape[0]} ROIs)",
        )


if __name__ == "__main__":
    main()

"""
topology_metrics.py  —  shared topological-metric computation.

Single source of truth imported by both `extract_schaefer_metrics.py`
(human connectomes) and `compute_rnn_topology.py` (model networks), so the two
sides are guaranteed to be processed identically.

Pipeline per network:
    1. prune to a target density (strongest edges),
    2. mean-normalize the surviving weights (scale-invariance),
    3. compute the metrics below.

Metrics returned
----------------
modularity / modularity_louvain : Newman Q of the weighted Louvain partition
                                  (networkx; kept under `modularity` for
                                  backward compatibility with existing CSVs).
modularity_leiden               : Newman Q of the weighted Leiden partition
                                  (leidenalg, ModularityVertexPartition). Same
                                  quality function as Louvain, different
                                  optimiser, so the two columns are directly
                                  comparable.
efficiency                      : weighted global efficiency (bct.efficiency_wei,
                                  length = 1/weight).
rich_club                       : WEIGHTED rich club, normalized against an
                                  ENSEMBLE of degree-preserving nulls and
                                  summarised as the mean of phi_norm(k) over the
                                  well-sampled degree range (trims the unstable
                                  high-k tail). Replaces the old
                                  single-null / max-over-k estimate.
small_worldness                 : Telesford omega = L_rand/L - C/C_latt on the
                                  binary backbone (bounded ~[-1, 1]).
participation                   : mean participation coefficient over nodes,
                                  using the Leiden modules (bct.participation_coef).

Null-model sizes are configurable below — they dominate runtime.
"""

import numpy as np
import networkx as nx
import bct
import leidenalg
import igraph as ig

# --- null-model / summary configuration (tune for speed vs precision) --------
N_NULL_RC    = 100   # randomized nulls for rich-club normalization
N_NULL_SW    = 10    # randomized nulls for small-worldness (path length)
N_LATT_SW    = 3     # lattice nulls for small-worldness (clustering)
RAND_ITER    = 5     # rewiring passes per edge in bct.randmio_und / latmio_und
RC_MIN_NODES = 10    # smallest rich-club size to trust (k range cut-off)
SEED         = 42

TARGET_DENSITY = 0.135


# =============================================================================
# Pruning / preparation
# =============================================================================
def prune_network(adj_matrix, target_density=TARGET_DENSITY):
    """Keep only the strongest edges to reach `target_density`."""
    adj = adj_matrix.copy()
    np.fill_diagonal(adj, 0)
    weights = adj[np.triu_indices_from(adj, k=1)]
    if len(weights) == 0:
        return adj, 0.0
    n_keep = int(len(weights) * target_density)
    sorted_w = np.sort(weights)[::-1]
    thr = sorted_w[n_keep] if n_keep < len(sorted_w) else 0.0
    adj[adj < thr] = 0
    return adj, float(thr)


def _prepare(adj_matrix, density_override=None):
    """Prune to density and mean-normalize weights; return (W_norm, W_bin, density)."""
    n_nodes = adj_matrix.shape[0]
    n_possible = n_nodes * (n_nodes - 1) / 2
    n_edges = np.count_nonzero(np.triu(adj_matrix, k=1))
    current_density = n_edges / n_possible if n_possible > 0 else 0.0

    target_d = density_override if density_override else TARGET_DENSITY
    if 0 < current_density < target_d:          # sparser than target: keep as-is
        target_d = current_density
        pruned = adj_matrix.copy()
        np.fill_diagonal(pruned, 0)
    else:
        pruned, _ = prune_network(adj_matrix, target_d)

    active = pruned[pruned > 0]
    W_norm = pruned / active.mean() if len(active) > 0 else pruned.copy()
    W_bin = (pruned > 0).astype(float)
    return W_norm, W_bin, float(target_d)


# =============================================================================
# Community structure
# =============================================================================
def _leiden_partition(W_norm, seed=SEED):
    """Weighted Leiden partition -> (list-of-sets, 1-indexed membership vector)."""
    n = W_norm.shape[0]
    src, tgt = np.triu_indices(n, k=1)
    mask = W_norm[src, tgt] > 0
    g = ig.Graph(n=n, edges=list(zip(src[mask].tolist(), tgt[mask].tolist())))
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition,
        weights=W_norm[src[mask], tgt[mask]].tolist(), seed=seed)
    return [set(c) for c in part], np.asarray(part.membership) + 1


# =============================================================================
# Weighted rich club, ensemble-normalized
# =============================================================================
def _rich_club_norm(W_norm, W_bin, n_null=N_NULL_RC, seed=SEED):
    """Mean of phi_norm(k) over the well-sampled degree range."""
    deg = W_bin.sum(0).astype(int)
    # bct.rich_club_wu divides by zero at high-k empty clubs (-> nan, which we
    # filter below); silence the resulting numpy warning to keep logs clean.
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = np.asarray(bct.rich_club_wu(W_norm), dtype=float)
        kmax = len(rc)
        if kmax == 0:
            return np.nan

        rng = np.random.default_rng(seed)
        acc = np.zeros(kmax)
        cnt = np.zeros(kmax)
        for _ in range(n_null):
            R, _ = bct.randmio_und(W_norm, RAND_ITER, seed=int(rng.integers(1e9)))
            rcr = np.asarray(bct.rich_club_wu(R), dtype=float)
            L = min(kmax, len(rcr))
            ok = np.isfinite(rcr[:L]) & (rcr[:L] > 0)
            acc[:L][ok] += rcr[:L][ok]
            cnt[:L][ok] += 1

        rc_rand = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1), np.nan)
        phi = rc / rc_rand

    ks = np.arange(1, kmax + 1)
    n_above = np.array([(deg > k).sum() for k in ks])
    valid = np.isfinite(phi) & (n_above >= RC_MIN_NODES)
    if valid.sum() < 2:
        valid = np.isfinite(phi)
        if valid.sum() == 0:
            return np.nan
        return float(np.nanmean(phi[valid]))
    kk, pp = ks[valid], phi[valid]
    return float(np.trapz(pp, kk) / (kk[-1] - kk[0]))


# =============================================================================
# Small-worldness (Telesford omega), binary backbone
# =============================================================================
def _small_worldness(W_bin, n_null=N_NULL_SW, n_latt=N_LATT_SW, seed=SEED):
    C = float(bct.clustering_coef_bu(W_bin).mean())
    L = float(bct.charpath(bct.distance_bin(W_bin), include_infinite=False)[0])
    if not np.isfinite(L) or L <= 0:
        return np.nan

    rng = np.random.default_rng(seed + 1)
    Lr = []
    for _ in range(n_null):
        R, _ = bct.randmio_und(W_bin, RAND_ITER, seed=int(rng.integers(1e9)))
        lr = bct.charpath(bct.distance_bin(R), include_infinite=False)[0]
        if np.isfinite(lr) and lr > 0:
            Lr.append(lr)
    Cl = []
    for _ in range(n_latt):
        Rlat = bct.latmio_und(W_bin, RAND_ITER, seed=int(rng.integers(1e9)))[0]
        Cl.append(float(bct.clustering_coef_bu(Rlat).mean()))

    if not Lr or not Cl or np.mean(Cl) <= 0:
        return np.nan
    return float(np.mean(Lr) / L - C / np.mean(Cl))


# =============================================================================
# Top-level
# =============================================================================
def compute_topology(adj_matrix, density_override=None,
                     n_null_rc=N_NULL_RC, n_null_sw=N_NULL_SW, seed=SEED):
    """Return a dict of topological metrics for one (weighted, symmetric) matrix."""
    W_norm, W_bin, density = _prepare(adj_matrix, density_override)
    G_w = nx.from_numpy_array(W_norm)

    # --- modularity: Louvain and Leiden, scored with the same quality fn -----
    try:
        louv = nx.community.louvain_communities(G_w, weight="weight", seed=seed)
        mod_louvain = nx.community.modularity(G_w, louv, weight="weight")
    except Exception:
        mod_louvain = np.nan
    try:
        leid, ci = _leiden_partition(W_norm, seed=seed)
        mod_leiden = nx.community.modularity(G_w, leid, weight="weight")
    except Exception:
        mod_leiden, ci = np.nan, None

    # --- weighted global efficiency (bctpy) ----------------------------------
    try:
        efficiency = float(bct.efficiency_wei(W_norm))
    except Exception:
        efficiency = np.nan

    # --- weighted, ensemble-normalized rich club -----------------------------
    try:
        rich_club = _rich_club_norm(W_norm, W_bin, n_null=n_null_rc, seed=seed)
    except Exception:
        rich_club = np.nan

    # --- small-worldness (omega) ---------------------------------------------
    try:
        small_world = _small_worldness(W_bin, n_null=n_null_sw, seed=seed)
    except Exception:
        small_world = np.nan

    # --- participation coefficient (Leiden modules) --------------------------
    # Isolated nodes (degree 0 after pruning) give a 0/0 inside bct; silence the
    # numpy warning and ignore any resulting nan when averaging over nodes.
    try:
        if ci is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                pc = bct.participation_coef(W_norm, ci)
            participation = float(np.nanmean(pc))
        else:
            participation = np.nan
    except Exception:
        participation = np.nan

    return {
        "density": density,
        "modularity": mod_louvain,            # backward-compatible name
        "modularity_louvain": mod_louvain,
        "modularity_leiden": mod_leiden,
        "efficiency": efficiency,
        "rich_club": rich_club,
        "small_worldness": small_world,
        "participation": participation,
    }

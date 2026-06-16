"""
Script: compute_rnn_topology.py
Environment: cluster pregnm39 / local ngym39 (requires networkx, bctpy, leidenalg, python-igraph)

Goal:
1. Load `weights_evolution.npy` for each run.
2. Prune to target density, mean-normalize weights.
3. Compute topological metrics (modularity [Louvain + Leiden], weighted
   efficiency, ensemble-normalized weighted rich club, small-worldness omega,
   participation) via the shared `topology_metrics` module.
4. Save results to `rnn_topological_metrics.csv`.

NOTE: the metric definitions live in `topology_metrics.py`, shared with
`extract_schaefer_metrics.py`, so human and model networks are processed
identically. The rich-club ensemble null makes this ~3 s/network; tune
`topology_metrics.N_NULL_RC` for speed.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from analysis.utils.paths import resolve_sweep_dir
from topology_metrics import compute_topology

import argparse


# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description="Compute RNN Topological Metrics")
    parser.add_argument("--sweep_dir", type=str, required=True, help="Path to sweep directory (e.g., logs/sweep/forage_v5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs (default: None = all)")
    return parser.parse_args()

OUTPUT_DIR = Path("analysis/Network_analysis/Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(adj_matrix, run_id="Unknown", density_override=None):
    """Thin wrapper around the shared topology pipeline (adds the run id)."""
    m = compute_topology(adj_matrix, density_override=density_override)
    m["run_id"] = run_id
    return m


def main():
    args = parse_args()

    try:
        logs_dir = resolve_sweep_dir(args.sweep_dir)
    except Exception as e:
        print(f"Error resolving sweep directory: {e}")
        return

    print(f">>> Scanning {logs_dir}...")
    runs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    runs.sort()

    if args.limit:
        selected_runs = runs[:args.limit]
        print(f"Limit applied: Processing first {len(selected_runs)} runs...")
    else:
        selected_runs = runs
        print(f"Processing all {len(selected_runs)} runs...")

    results = []

    for i, run_dir in enumerate(selected_runs):
        npy_path = run_dir / "weights_evolution.npy"

        if not npy_path.exists():
            print(f"Skipping {run_dir.name}: No weights_evolution.npy")
            continue

        print(f"Processing [{i+1}/{len(selected_runs)}]: {run_dir.name}")

        try:
            # Load [T, N, N]
            W_evol = np.load(npy_path)
            # Take last time point
            W_last = W_evol[-1]

            # Make W_rec comparable to the (symmetric, positive) human connectomes:
            #   1. take the magnitude of each weight (excitatory & inhibitory -> strength)
            #   2. symmetrise, so i->j and j->i have the same value
            W_last = np.abs(W_last)
            W_last = (W_last + W_last.T) / 2.0

            m = compute_metrics(W_last, run_id=run_dir.name)
            m['path'] = str(run_dir)

            results.append(m)
            print(f"  > Mod(Louvain/Leiden): {m.get('modularity_louvain', float('nan')):.3f}"
                  f"/{m.get('modularity_leiden', float('nan')):.3f}, "
                  f"Eff: {m.get('efficiency', float('nan')):.3f}, "
                  f"RC: {m.get('rich_club', float('nan')):.3f}, "
                  f"omega: {m.get('small_worldness', float('nan')):.3f}")

        except Exception as e:
            print(f"  > Error processing {run_dir.name}: {e}")

    # Save
    if not results:
        print("No results computed.")
        return

    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "rnn_topological_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

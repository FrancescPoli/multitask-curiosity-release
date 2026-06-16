"""
Script: extract_schaefer_metrics.py
Environment: ngym39 (requires h5py, netneurotools, bctpy, leidenalg, python-igraph)

Goal: 
1. Load `harmonized_connectomes.mat` (2419 Schaefer 100x100 subjects)
2. Load `age.mat` and assign dataset labels (HCPd, HCPya, HCPa)
3. Compute Consensus Network (using HCPya 26-35 subset with distance-dependence)
4. Compute Topological Metrics (Consensus + all 2419 individual subjects)
5. Save results to `human_topological_metrics.csv`
"""

import numpy as np
import scipy.io
import h5py
import pandas as pd
import sys
from netneurotools.networks import struct_consensus
from pathlib import Path
import os
import time

sys.path.append(str(Path(__file__).resolve().parent))
from topology_metrics import compute_topology

# --- Configuration ---
DATA_DIR = Path("analysis/Network_analysis/Data/Atlases/Schaefer")
OUTPUT_DIR = Path("analysis/Network_analysis/Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAT_FILE = DATA_DIR / "harmonized_connectomes.mat"
AGE_FILE = DATA_DIR / "age.mat"
DIST_FILE = Path("brain/assets/euclidean_distances_schaefer100.npy")

# --- Helper Functions ---

def compute_metrics(adj_matrix, subject_id="Unknown", exact_age=np.nan, dataset="Unknown", density_override=None):
    """Thin wrapper around the shared topology pipeline (adds subject metadata)."""
    m = compute_topology(adj_matrix, density_override=density_override)
    m["full_id"] = subject_id
    m["dataset"] = dataset
    m["age"] = exact_age
    return m

# --- Main ---

def main():
    print(">>> Loading Data...")
    
    # 1. Distances and HemiID
    try:
        euclidean_distances = np.load(DIST_FILE)
        # Schaefer 100 perfectly maps: 1-50 Left, 51-100 Right
        hemiid = np.array([0]*50 + [1]*50).reshape(-1, 1)
        print("Successfully built euclidean distances and hemi maps.")
    except Exception as e:
        print(f"Failed to load distance atlas: {e}")
        return

    # 2. Load Ages
    print(f"Loading Ages: {AGE_FILE}")
    try:
        with h5py.File(AGE_FILE, 'r') as f:
            ages = f['age'][()].flatten()  # Expected 2419
    except Exception as e:
        print(f"Failed to load age file: {e}")
        return

    # 3. Load Connectome Data
    print(f"Loading Connectomes: {MAT_FILE}")
    try:
        with h5py.File(MAT_FILE, 'r') as f:
            all_connectomes = f['harmonized_connectomes'][()]  # Expected (100, 100, 2419)
    except Exception as e:
        print(f"Failed to load connectomes: {e}")
        return
        
    print(f"Loaded connectomes shape: {all_connectomes.shape}")
    n_subjects = all_connectomes.shape[2]
    
    if len(ages) != n_subjects:
        print(f"Mismatch: {n_subjects} connectomes vs {len(ages)} ages!")
        return

    # Create dataset mapping based on known indices
    datasets = []
    for i in range(n_subjects):
        if i <= 635:
            datasets.append("HCPd")
        elif i <= 1700:
            datasets.append("HCPya")
        else:
            datasets.append("HCPa")
    datasets = np.array(datasets)

    # 3. Extract consensus cohort (HCPya 26-35)
    consensus_indices = np.where((datasets == "HCPya") & (ages >= 26) & (ages <= 35))[0]
    print(f"Found {len(consensus_indices)} subjects for the Golden Adult Consensus (HCPya 26-35y)")

    if len(consensus_indices) > 0:
        consensus_connectomes = all_connectomes[:, :, consensus_indices]
        print(">>> Computing Consensus Network...")
        try:
            consensus_mat = struct_consensus(consensus_connectomes, euclidean_distances, hemiid, weighted=True)
            print("Computing metrics for Consensus...")
            cons_metrics = compute_metrics(consensus_mat, subject_id="HCPya_Consensus_26-35", exact_age=np.nan, dataset="Consensus")
            print("Consensus Metrics:", cons_metrics)
        except Exception as e:
            print(f"Consensus computation failed: {e}")
            cons_metrics = None
    else:
        print("Skipping consensus, no subjects matched.")
        cons_metrics = None
    
    # 4. Compute Indiv Metrics for ALL
    print(f">>> Computing Individual Metrics for all {n_subjects} subjects across the lifespan...")
    results = []
    
    if cons_metrics:
        results.append(cons_metrics)
    
    t0 = time.time()
    for i in range(n_subjects):
        if datasets[i] == "HCPya":
            subj_mat = all_connectomes[:, :, i]
            m = compute_metrics(subj_mat, subject_id=f"Subj_{i}", exact_age=ages[i], dataset=datasets[i])
            results.append(m)
            
            if len(results) % 200 == 0:
                print(f"Processed {len(results)} HCPya subjects...")

    print(f"Finished metrics calculation in {time.time() - t0:.2f} seconds.")

    # 5. Save
    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "human_topological_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved comprehensive human baseline results to {out_path}")

if __name__ == "__main__":
    main()

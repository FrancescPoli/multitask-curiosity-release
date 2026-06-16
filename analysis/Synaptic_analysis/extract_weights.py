"""
Batch Human Comparison Analysis
===============================

Iterates through a sweep directory, extracts weight evolution for all models,
parses hyperparameters, and aggregates them into a single population CSV.

Changes:
- Supports resuming from existing CSV.
- Saves incrementally after each run.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd
import re

# Add project root to path for imports
# parent = Synaptic_analysis, parent.parent = analysis, parent.parent.parent = project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# CORRECTED IMPORT
from analysis.extract_weights_evolution import extract_weight_evolution
from analysis.utils.paths import resolve_sweep_dir

def parse_params_from_dirname(dirname):
    """
    Extracts L1, Alpha, Beta, Temp, Travel from directory name.
    Example: run_031_l1-1e-05_beta0.003_alpha0.01_trav50_blk1_temp1.0
    """
    params = {}
    
    # Regex patterns
    patterns = {
        'l1': r'l1-([\d\.e-]+)',
        'beta': r'beta([\d\.e-]+)',
        'alpha': r'alpha([\d\.e-]+)',
        'travel': r'trav(\d+)',
        'temp': r'temp([\d\.e-]+)',
        'eps': r'eps(-?[\d\.e-]+)',
        'seed': r'run_(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, dirname)
        if match:
            val_str = match.group(1)
            try:
                if key == 'travel' or key == 'seed':
                    params[key] = int(val_str)
                else:
                    params[key] = float(val_str)
            except ValueError:
                params[key] = None
        else:
            # eps is omitted from dirname when 0.0 — default to 0.0, not None
            params[key] = 0.0 if key == 'eps' else None

    return params


def main():
    parser = argparse.ArgumentParser(description="Batch Extract Weights for Population Analysis")
    parser.add_argument("--sweep_dir", type=str, required=True, help="Path to sweep directory (e.g., logs/sweep/forage_v5)")
    parser.add_argument("--output", type=str, default="analysis/Synaptic_analysis/Data/population_weights.csv", help="Output CSV file")
    parser.add_argument("--save_npy", action="store_true", help="Save full weight matrices to weights_evolution.npy in run dir")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs (for testing)")
    
    args = parser.parse_args()
    
    try:
        sweep_dir = resolve_sweep_dir(args.sweep_dir)
        output_path = Path(args.output)
    except Exception as e:
        print(f"Error resolving sweep directory: {e}")
        sys.exit(1)
    
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        sys.exit(1)
        
    runs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    runs.sort()
    
    if args.limit:
        runs = runs[:args.limit]
        
    print(f"Found {len(runs)} runs in {sweep_dir}")
    
    # Check for existing results to resume
    processed_ids = set()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            if 'run_id' in existing_df.columns:
                processed_ids = set(existing_df['run_id'].unique())
                print(f"Resuming: Found {len(processed_ids)} runs already processed in {output_path}")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    total_runs = len(runs)
    processed_count = 0

    for i, run_dir in enumerate(runs):
        # Determine if we skip
        # If saving npy, we might want to force if npy doesn't exist even if csv exists
        skip = False
        cached_file = run_dir / "weights_evolution.csv"
        npy_file = run_dir / "weights_evolution.npy"
        
        if run_dir.name in processed_ids:
            # Already in main csv, but do we need npy?
            if args.save_npy and not npy_file.exists():
                skip = False # Need to process to get npy
            else:
                skip = True
        
        if skip:
            continue

        print(f"[{i+1}/{total_runs}] Processing {run_dir.name}...")
        
        try:
            # 1. Extract from checkpoints
            # We always extract if save_npy is requested and missing, or if not cached
            if args.save_npy and not npy_file.exists():
                 print(f"  > Extracting weights (saving matrices)...")
                 df = extract_weight_evolution(run_dir, save_matrices=True)
            elif not cached_file.exists():
                 print(f"  > Extracting weights...")
                 df = extract_weight_evolution(run_dir, save_matrices=args.save_npy)
            else:
                 # Cache exists, and we don't need npy or it exists
                 df = pd.read_csv(cached_file)
            
            # Cache it for future use if it was computed (df not empty)
            if not df.empty and not cached_file.exists():
                df.to_csv(cached_file, index=False)
            
            if df.empty:
                print(f"  > No data found/extracted.")
                continue
                
            # 3. Add Metadata
            params = parse_params_from_dirname(run_dir.name)
            for key, val in params.items():
                df[key] = val
            
            df['run_id'] = run_dir.name
            
            # 4. Save Incrementally to Main CSV
            # Only append if not already in processed_ids to avoid duplicates
            if run_dir.name not in processed_ids:
                file_exists = output_path.exists()
                df.to_csv(output_path, mode='a', header=not file_exists, index=False)
                processed_ids.add(run_dir.name)
                processed_count += 1
            
        except Exception as e:
            print(f"Failed to process {run_dir.name}: {e}")
            continue

    if processed_count == 0:
        print("No new data added to main CSV.")
    else:
        print(f"\nBatch complete! Added {processed_count} new runs. Total combined in {output_path}")

if __name__ == "__main__":
    main()

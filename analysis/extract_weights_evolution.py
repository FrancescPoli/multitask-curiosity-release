"""
Human Brain Comparison Analysis
===============================

Extracts model weight evolution (L1 Sum) and compares it with human synaptic density data.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# Ensure analysis folder is in path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.plotting.utils import (
    get_Wrec,
    list_checkpoint_steps_and_paths,
    build_model_from_meta,
    load_state_into_model,
    load_meta
)

def extract_weight_evolution(run_dir: Path, save_matrices: bool = False, device: torch.device = torch.device("cpu")) -> pd.DataFrame:
    """
    Extract L1 Sum of Recurrent Weights (W_rec) over training steps.
    If save_matrices is True, saves the full [T, N, N] weight evolution to weights_evolution.npy in run_dir.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"Processing run: {run_dir.name}")

    # 1. Load Meta
    meta = load_meta(run_dir)
    
    # 2. List Checkpoints
    ckpts = list_checkpoint_steps_and_paths(run_dir)
    if not ckpts:
        print(f"Warning: No checkpoints found in {run_dir}")
        return pd.DataFrame()

    print(f"Found {len(ckpts)} checkpoints.")

    # 3. Build Model
    model = build_model_from_meta(meta, device=device)

    records = []
    
    # Container for weight matrices if saving
    all_W_rec = []

    # 4. Iterate and Extract
    for step, path in ckpts:
        try:
            obj = torch.load(path, map_location=device)
            state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
            
            load_state_into_model(model, state)
            W_rec_tensor = get_Wrec(model)
            W_rec = W_rec_tensor.detach()
            
            # metrics
            w_abs = W_rec.abs()
            l1_sum = w_abs.sum().item()
            l0_norm = (w_abs > 1e-5).sum().item() # Threshold for "non-zero"
            
            records.append({
                "step": step,
                "l1_sum": l1_sum,
                "l0_norm": l0_norm,
                "run_id": run_dir.name
            })
            
            if save_matrices:
                # Store as numpy (cpu)
                all_W_rec.append(W_rec.cpu().numpy())
                
        except Exception as e:
            print(f"Error reading checkpoint {path.name}: {e}")
            continue

    df = pd.DataFrame(records)
    
    if save_matrices and all_W_rec:
        try:
            # Stack to [T, N, N]
            W_stack = np.stack(all_W_rec)
            save_path = run_dir / "weights_evolution.npy"
            np.save(save_path, W_stack)
            print(f"  > Saved weight matrices shape {W_stack.shape} to {save_path}")
        except Exception as e:
            print(f"  > Error saving weight matrices: {e}")
            
    return df

def main():
    parser = argparse.ArgumentParser(description="Extract Model Weight Evolution for Human Comparison")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the model run directory")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file (default: rights_evolution.csv in run_dir)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "weights_evolution.csv"
    
    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        sys.exit(1)
        
    df = extract_weight_evolution(run_dir)
    
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"\nSuccess! Weight evolution saved to: {output_path}")
        print(df.head())
    else:
        print("\nNo data extracted.")

if __name__ == "__main__":
    main()

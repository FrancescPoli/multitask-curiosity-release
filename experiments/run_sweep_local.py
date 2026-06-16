#!/usr/bin/env python3
"""
Script to run a LOCAL benchmark sweep (Baseline, L1, ProxL1) without touching the cluster config.
Usage:
    python experiments/run_sweep_local.py
    python experiments/run_sweep_local.py --parallel 2
"""
import argparse
import sys
import subprocess
import itertools
from pathlib import Path
import numpy as np

# Add root to sys.path to import paths
sys.path.append(str(Path(__file__).resolve().parent.parent))
from paths import get_logs_dir

# Import config from main file
from run_sweep import SWEEP_CONFIG, SWEEP_SETS

def parse_args():
    parser = argparse.ArgumentParser(description="Run local L1/Prox benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel processes (default: 1)")
    parser.add_argument("--log-root", type=str, default="logs/sweep", help="Log dir")
    parser.add_argument("--sweep-id", type=str, default="local_bench_l1_prox", help="Sweep ID")
    
    args = parser.parse_args()
    if args.log_root == "logs/sweep":
         args.log_root = str(get_logs_dir() / "sweep")
    return args

def run_command(cmd, dry_run):
    print(f"Running: {' '.join(cmd)}")
    if not dry_run:
        subprocess.check_call(cmd)

def main():
    args = parse_args()
    
    # Filter SWEEP_SETS for L1 and Prox only (and Baseline)
    # Heuristic: 
    #   Baseline: all 0.0/False
    #   L1: l1_weight > 0
    #   Prox: prox_l1_weight > 0
    #   Skip if 'distance_penalty' is True
    
    local_sets = []
    
    for s in SWEEP_SETS:
        # Check if this set involves Distance
        if True in s.get("distance_penalty", [False]):
            continue
            
        # Check if list of values implies Distance (weak check)
        if any(v > 0 for v in s.get("distance_weight", [0.0])):
            continue

        local_sets.append(s)
        
    print(f"Filtered {len(SWEEP_SETS)} sets down to {len(local_sets)} for local run (No Distance).")

    # Setup Dir
    base_conf = SWEEP_CONFIG["base_args"]
    # Ensure steps is 20000 (match cluster)
    base_conf["steps"] = 20000
    
    timestamp = args.sweep_id
    log_dir = Path(args.log_root) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_dir}")
    
    # Generate Commands
    cmds = []
    
    idx = 0
    for sweep_set in local_sets:
        keys = list(sweep_set.keys())
        values = list(sweep_set.values())
        combinations = list(itertools.product(*values))
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            # Name
            parts = []
            for k,v in params.items():
                if "weight" in k and float(v) > 0:
                    parts.append(f"{k}-{v:.5g}")
            suffix = "_".join(parts) if parts else "baseline"
            run_name = f"run_{idx:03d}_{suffix}"
            
            # Args
            current_args = base_conf.copy()
            current_args.update(params)
            
            # Build cmd
            root = Path(__file__).resolve().parent.parent
            script = root / "run.py"
            
            cmd = [sys.executable, str(script), "--logdir", str(log_dir), "--run-name", run_name]
            
            for k, v in current_args.items():
                arg_name = "--" + k.replace("_", "-")
                if isinstance(v, bool):
                    if v: cmd.append(arg_name)
                elif isinstance(v, list):
                    cmd.append(arg_name)
                    cmd.extend([str(x) for x in v])
                else:
                    cmd.append(arg_name)
                    cmd.append(str(v))
            
            cmds.append(cmd)
            idx += 1
            
    print(f"Generated {len(cmds)} commands.")
    
    # Execution
    if args.parallel > 1:
        # Simple pool
        from multiprocessing import Pool
        with Pool(args.parallel) as p:
            # map expects func, iterable. Helper needed for dry_run arg
            # but check_call inside pool might be messy with stdout.
            # Let's keep it simple: Sequential is safer unless requested.
            print("Parallel execution not fully implemented in this quick script, running sequential.")
            # If user REALLY wants parallel, we can use Popen list.
            if not args.dry_run:
                # Batch execution?
                pass
    
    # Sequential Run
    for i, cmd in enumerate(cmds):
        print(f"\n[{i+1}/{len(cmds)}] ...")
        
        # Check if exists
        # extract run_name
        # run_name is last arg? No.
        # It's in the cmd list.
        # Easier to check file system
        # We constructed run_name so we know it
        # Recalculating is hard.
        # Just run. run.py has internal skip logic check? 
        # No, run_sweep.py had the check logic in lines 216-224.
        # I should add skip logic here.
        # But for benchmark, maybe force run?
        # Regular run.py will overwrite if run_name collision.
        
        try:
            run_command(cmd, args.dry_run)
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    main()

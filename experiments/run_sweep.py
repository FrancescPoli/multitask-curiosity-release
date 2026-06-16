#!/usr/bin/env python3
"""
Script to run parameter sweeps for Multitask Curiosity.
Acts as the Single Source of Truth for experiment configurations.

Usage:
    python experiments/run_sweep.py --dry-run
    python experiments/run_sweep.py
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import List
import numpy as np
import datetime

# Add root to sys.path to import paths
sys.path.append(str(Path(__file__).resolve().parent.parent))
from paths import get_logs_dir

# =============================================================================
# SWEEP CONFIGURATION (SINGLE SOURCE OF TRUTH)
# =============================================================================


# =============================================================================
# SWEEP CONFIGURATIONS
# =============================================================================

# 1. YANG OG (Exact Replication)
# - Hidden=256
# - Only Baseline and L1 sweep
# - No distance penalty
SWEEP_CONFIG_YANG_OG = {
    "base_args": {
        # Training Dynamics
        "steps": 312500,         # 20M trials (match user capacity)
        "batch_size": 64,
        "log_every": 3125,
        "save_every": 15625,
        "scaffolding": False,
        "seed": 0,
        
        # Model (Yang 2019 Exact)
        "hidden": 256,
        "activation": "relu",
        "w_rec_init": "randortho",
        "use_separate_input": False, # Fused input
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",

        # Tasks (All 20 - Poli equivalents of Yang19)
        "tasks": [
            'poli.go', 'poli.rtgo', 
            'poli.dlygo',
            'poli.antigo', 'poli.rtantigo', 
            'poli.dlyantigo',
            'poli.dm1', 'poli.dm2', 'poli.multidm', 'poli.multidlydm',
            'poli.ctxdm1', 'poli.ctxdm2', 'poli.ctxdlydm1', 'poli.ctxdlydm2',
            'poli.dlydm1', 'poli.dlydm2',
            'poli.dlyms', 'poli.dlynms', 'poli.catdlyms', 'poli.catdlynms'
        ],
    },
    "grid": [
        # Baseline
        {"l1_weight": [0.0], "distance_penalty": [False]},
        # L1 Sweep (10 models total)
        # Log-spaced: 9 values from 1e-9 to 1e-4 (inclusive)
        {"l1_weight": list(np.logspace(-9, -4, 9)), "distance_penalty": [False]}
    ]
}

# 2. DEFAULT (Distance + L1 Sweep)
# - Hidden=100 (Matches Schaefer atlas 100 regions)
# - Full sweep: Baseline, L1, Distance, L1+Distance
SWEEP_CONFIG_DEFAULT = {
    "base_args": {
        # Same Training Dynamics
        "steps": 312500,
        "batch_size": 64,
        "log_every": 3125,
        "save_every": 15625,
        "scaffolding": False,
        "seed": 0,
        
        # Model (Adapted for Brain Distance)
        "hidden": 100,           # Matches Schaefer 100 parcellation
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,

        # Tasks
        "tasks": SWEEP_CONFIG_YANG_OG["base_args"]["tasks"],

        # Biology
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
    },
    "grid": [
        # 1. Baseline (1 model)
        {
            "l1_weight": [0.0],
            "distance_penalty": [False], 
            "distance_weight": [0.0],
        },
        
        # 2. L1 Only (9 models) -> Total 10 (with baseline)
        {
            "l1_weight": list(np.logspace(-9, -4, 9)), 
            "distance_penalty": [False],
            "distance_weight": [0.0],
        },
        
        # 3. Distance Only (8 models) -> Total 8
        # Spanning sensible range for distance penalty
        # 1000 is high but plausible if we want to enforce extreme locality
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": list(np.logspace(-4, 3, 8)), # 1e-4 to 1000
        },

        # 4. Joint Grid (9x8 = 72 models)
        {
            "l1_weight": list(np.logspace(-9, -3, 9)),
            "distance_penalty": [True],
            "distance_weight": list(np.logspace(-9, -3, 8)),
        }
    ]
}

CONFIGS = {
    "default": SWEEP_CONFIG_DEFAULT,
    "yang_og": SWEEP_CONFIG_YANG_OG
}

# =============================================================================
# EXECUTION LOGIC
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run parameter sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--log-root", type=str, default="logs/sweep", help="Root directory for sweep logs")
    parser.add_argument("--sweep-id", type=str, default=None, help="Optional name for the sweep folder")
    parser.add_argument("--config", type=str, default="default", choices=["default", "yang_og"], help="Choose sweep configuration")
    args = parser.parse_args()
    if args.log_root == "logs/sweep":
         args.log_root = str(get_logs_dir() / "sweep")
    return args

def run_command(cmd: List[str], dry_run: bool):
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    if not dry_run:
        subprocess.check_call(cmd)

def main():
    args = parse_args()
    
    # Select Config
    selected_config = CONFIGS[args.config]
    base_conf = selected_config["base_args"]
    sweep_sets = selected_config["grid"]
    
    if args.sweep_id:
        timestamp = args.sweep_id
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    args.log_root = str(Path(args.log_root) / timestamp / args.config) # Nest under config name
    print(f"Sweep Logs will be saved to: {args.log_root}")
    
    # Generate Configs
    configs = []
    global_idx = 0
    
    for sweep_set in sweep_sets:
        keys = list(sweep_set.keys())
        values = list(sweep_set.values())
        combinations = list(itertools.product(*values))
        
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            current = base_conf.copy()
            current.update(param_dict)
            
            # Smart Naming Logic
            parts = []
            for k, v in param_dict.items():
                if "weight" in k and float(v) > 0:
                     parts.append(f"{k}-{v:.5g}")
            
            if not parts: suffix = "baseline"
            else: suffix = "_".join(parts)
            
            run_name = f"run_{global_idx:03d}_{suffix}"
            current["_run_name"] = run_name
            configs.append(current)
            global_idx += 1

    print(f"Total configurations: {len(configs)}")
    
    # Serial Execution Loop
    for i, conf in enumerate(configs):
        run_name = conf["_run_name"]
        
        root_dir = Path(__file__).resolve().parent.parent
        run_script = root_dir / "run.py"
        
        cmd = [sys.executable, str(run_script)]
        cmd.extend(["--logdir", args.log_root])
        cmd.extend(["--run-name", run_name])
        
        for k, v in conf.items():
            if k.startswith("_"): continue # Skip internal meta-keys
            arg_name = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                # Only add flag if True; False relies on run.py defaults
                if v:
                    cmd.append(arg_name)
            elif isinstance(v, list):
                cmd.append(arg_name)
                cmd.extend([str(item) for item in v])
            else:
                cmd.append(arg_name)
                cmd.append(str(v))
        
        print(f"\n--- Run {i+1}/{len(configs)} : {run_name} ---")
        try:
            run_command(cmd, args.dry_run)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to run parameter sweeps for Multitask Curiosity (Foraging Edition).
Acts as the Single Source of Truth for experiment configurations.

Usage:
    python experiments/run_sweep_forage.py --dry-run
    python experiments/run_sweep_forage.py --config forage
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
# SWEEP CONFIGURATIONS
# =============================================================================

# Common tasks (Poli suite)
# Common tasks (Poli suite - Compositional Training Split: 20 Tasks)
POLI_TASKS = [
    # 1. Reach (10 Tasks)
    'poli.go', 'poli.rtgo', 
    'poli.dlygo', 'poli.antigo', 'poli.ctxgo',
    'poli.dlyantigo', 'poli.dlyctxgo', 'poli.antictxgo', 'poli.dlyantictxgo', 'poli.rtantictxgo',
    
    # 2. Decision (6 Tasks - No Context)
    'poli.dm1', 'poli.dm2', 
    'poli.dlydm1', 'poli.multidm', 
    'poli.antidm1', 'poli.antidlydm1',

    # 3. Match (4 Tasks - No Anti/Ctx)
    'poli.dlyms', 'poli.dlynms', 'poli.catdlyms', 'poli.catdlynms'
]

SWEEP_CONFIG_FORAGE = {
    "base_args": {
        # Training Dynamics
        "steps": 312500,         # 20M trials
        "batch_size": 1,
        "log_every": 3125,
        "save_every": 15625,
        "scaffolding": False,
        "seed": 0,
        
        # Model (Schaefer 100 style)
        "hidden": 100,
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,

        # Tasks
        "tasks": POLI_TASKS,

        # Biology (paths)
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
        
        # Foraging Defaults
        "foraging": "mvt",
        "forage_alpha_local": 0.03,      # fast
        "forage_min_block_steps": 50,    # stable blocks
        "forage_eps": 0.0,
        "forage_temperature": 0.0,       # deterministic for now
    },
    "grid": [
        # CROSS PRODUCT: 4 Reg States x 18 Forage States = 72 runs
        
        # Grid variables:
        # beta_global: [0.001, 0.003, 0.01]
        # travel_steps: [50, 100, 200]
        # temp: [0.0, 0.05]  (Deterministic vs Stochastic)

        # 1. Baseline (No Reg)
        {
            "l1_weight": [0.0],
            "distance_penalty": [False],
            "foraging": ["mvt"],
            "forage_beta_global": [0.001, 0.003, 0.01],
            "forage_travel_steps": [50, 100, 200],
            "forage_temperature": [0.0, 0.05],
        },

        # 2. L1 Only
        {
            "l1_weight": [1e-5],
            "distance_penalty": [False],
            "foraging": ["mvt"],
            "forage_beta_global": [0.001, 0.003, 0.01],
            "forage_travel_steps": [50, 100, 200],
            "forage_temperature": [0.0, 0.05],
        },

        # 3. Distance Only
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": [1e-4],
            "foraging": ["mvt"],
            "forage_beta_global": [0.001, 0.003, 0.01],
            "forage_travel_steps": [50, 100, 200],
            "forage_temperature": [0.0, 0.05],
        },

        # 4. Joint (L1 + Distance)
        {
            "l1_weight": [1e-5],
            "distance_penalty": [True],
            "distance_weight": [1e-4],
            "foraging": ["mvt"],
            "forage_beta_global": [0.001, 0.003, 0.01],
            "forage_travel_steps": [50, 100, 200],
            "forage_temperature": [0.0, 0.05],
        }
    ]
}

SWEEP_CONFIG_FORAGE_V2 = {
    "base_args": {
        "steps": 312500,
        "batch_size": 1,
        "log_every": 3125,
        "save_every": 15625,
        "scaffolding": False,
        "seed": 0,
        "hidden": 100,
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,
        "tasks": POLI_TASKS,
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
        "foraging": "mvt",
        "forage_min_block_steps": 10,  # Reduced from 50
        "forage_eps": 0.0,
    },
    "grid": [
        # Grid Variables:
        # Beta Global: [0.003, 0.01, 0.03] (Shifted Higher)
        # Alpha Local: [0.03, 0.1, 0.3] (New Sweep)
        # Travel Steps: [10, 30, 50, 100] (New Sweep)
        # Temp: [0.0, 0.05]
        
        # 1. L1 Only (2 values)
        {
            "l1_weight": [1e-5, 1e-4],
            "distance_penalty": [False],
            "forage_beta_global": [0.001, 0.01, 0.1],
            "forage_alpha_local": [0.03, 0.1],
            "forage_travel_steps": [20, 50],
            "forage_temperature": [0.0, 0.01],
        },
        
        # 2. Distance Only (2 values)
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": [1e-4, 1e-3],
            "forage_beta_global": [0.001, 0.01, 0.1],
            "forage_alpha_local": [0.03, 0.1],
            "forage_travel_steps": [20, 50],
            "forage_temperature": [0.0, 0.01],
        }
    ]
}

SWEEP_CONFIG_FORAGE_V3 = {
    "base_args": {
        "steps": 400000, # 400k Steps (Batch=16) -> ~6.4M trials
        "batch_size": 16,
        "log_every": 10000,
        "save_every": 40000,
        "scaffolding": False,
        "seed": 0,
        "hidden": 100,
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,
        "tasks": POLI_TASKS,
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
        "foraging": "mvt",
        "forage_min_block_steps": 10,
        "forage_eps": 0.0,
    },
    "grid": [
        # Grid Variables:
        # Beta Global: [0.001, 0.005, 0.01]
        # Alpha Local: [0.01, 0.03] (Sticky vs Standard)
        # Travel Steps: [50, 500] (Standard vs High Cost)
        # Temp: [0.0, 0.005, 0.01]
        
        # 1. L1 Only
        {
            "l1_weight": [1e-5],
            "distance_penalty": [False],
            "forage_beta_global": [0.001, 0.005, 0.01],
            "forage_alpha_local": [0.01, 0.03],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1, 10],
            "forage_temperature": [0.0, 0.0001, 0.001],
        },
        
        # 2. Distance Only
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": [1e-4],
            "forage_beta_global": [0.001, 0.005, 0.01],
            "forage_alpha_local": [0.01, 0.03],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1, 10],
            "forage_temperature": [0.0, 0.0001, 0.001],
        }
    ]
}

SWEEP_CONFIG_FORAGE_V4 = {
    "base_args": {
        "steps": 400000, 
        "batch_size": 16,
        "log_every": 10000,
        "save_every": 40000,
        "scaffolding": False,
        "seed": 42, # NEW: Seed 42 for "messy" random projections
        "hidden": 100,
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,
        "tasks": POLI_TASKS,
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
        "foraging": "mvt",
        "forage_min_block_steps": 10,
        "forage_eps": 0.0,
        
        # Anti-Orthogonality
        "rule_encoding": "lowrank",
        "rule_dim_low": 4,
        "rule_dim_out": 100, # Reserve space for 100 tasks (Generalization ready)
    },
    "grid": [
        # Grid Variables:
        # Beta Global: [0.001, 0.005, 0.01]
        # Travel Steps: [50, 500] (Standard vs High Cost)
        # Temp: [0.0, 0.005] (Deterministic vs Stochastic)
        
        # 1. L1 Only
        {
            "l1_weight": [1e-5],
            "distance_penalty": [False],
            "forage_beta_global": [0.001, 0.005, 0.01],
            "forage_alpha_local": [0.01, 0.03],
            "forage_travel_steps": [50, 500],
            "forage_temperature": [0.0, 0.005],
        },
        
        # 2. Distance Only
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": [1e-4],
            "forage_beta_global": [0.001, 0.005, 0.01],
            "forage_alpha_local": [0.01, 0.03],
            "forage_travel_steps": [50, 500],
            "forage_temperature": [0.0, 0.005],
        },

        # 3. Base (Control)
        {
            "l1_weight": [0.0],
            "distance_penalty": [False],
            "forage_beta_global": [0.001, 0.005, 0.01],
            "forage_alpha_local": [0.01, 0.03],
            "forage_travel_steps": [50, 500],
            "forage_temperature": [0.0, 0.005],
        }
    ]
}

SWEEP_CONFIG_FORAGE_V5 = {
    "base_args": {
        "steps": 600000, 
        "batch_size": 16,
        "log_every": 10000,
        "save_every": 40000,
        "scaffolding": False,
        "seed": 42, # Messy Random Projections
        "hidden": 100,
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,
        "tasks": POLI_TASKS,
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
        "foraging": "mvt",
        "forage_eps": 0.0,
        
        # Anti-Orthogonality
        "rule_encoding": "lowrank",
        "rule_dim_low": 4,
        "rule_dim_out": 100,
    },
    "grid": [
        # Grid Variables:
        # Beta: [0.0001, 0.001, 0.003] (3)
        # Alpha: [0.01, 0.03, 0.1] (3)
        # Travel: [50, 500] (2)
        # Temp: [0.0001, 1.0] (2)
        # Min Block: [1] (Fixed)
        # Common Multiplier: 3*3*2*2 = 36 per reg value
        
        # 1. L1 Only (3 values -> 108 models)
        {
            "l1_weight": [1e-6, 1e-5, 1e-4],
            "distance_penalty": [False],
            "forage_beta_global": [0.0001, 0.001, 0.003],
            "forage_alpha_local": [0.01, 0.03, 0.1],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1],
            "forage_temperature": [0.0001, 1.0],
        },
        
        # 2. Distance Only (2 values -> 72 models)
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": [1e-5, 1e-4],
            "forage_beta_global": [0.0001, 0.001, 0.003],
            "forage_alpha_local": [0.01, 0.03, 0.1],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1],
            "forage_temperature": [0.0001, 1.0],
        },
    ]
}

SWEEP_CONFIG_FORAGE_V6 = {
    "base_args": {
        "steps": 600000, 
        "batch_size": 16,
        "log_every": 10000,
        "save_every": 40000,
        "scaffolding": False,
        "seed": 42, 
        "hidden": 100,
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,
        "tasks": POLI_TASKS,
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
        "foraging": "mvt",
        "forage_eps": 0.0,
        
        "rule_encoding": "lowrank",
        "rule_dim_low": 4,
        "rule_dim_out": 100,
    },
    "grid": [
        # Extended V5 Grid:
        # Beta: [0.0001, 0.001, 0.003] (3) - Kept Max 0.003
        # Alpha: [0.01, 0.03, 0.1] (3) - Keep
        # Travel: [50, 500] (2) - Keep
        # Temp: [0.0001, 0.001, 0.01, 1.0] (4) - Added 0.001, 0.01. Kept 1.0!
        # Eps: [0.0, -0.01] (2) - Added -0.01 (Stickiness)
        
        # 1. L1 Only
        {
            "l1_weight": [1e-6, 1e-5, 1e-4],
            "distance_penalty": [False],
            "forage_beta_global": [0.0001, 0.001, 0.003],
            "forage_alpha_local": [0.01, 0.03, 0.1],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1],
            "forage_temperature": [0.0001, 0.001, 0.01, 1.0],
            "forage_eps": [0.0, -0.01]
        },
        
        # 2. Distance Only
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": [1e-5, 1e-4],
            "forage_beta_global": [0.0001, 0.001, 0.003],
            "forage_alpha_local": [0.01, 0.03, 0.1],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1],
            "forage_temperature": [0.0001, 0.001, 0.01, 1.0],
            "forage_eps": [0.0, -0.01]
        },
    ]
}

SWEEP_CONFIG_FORAGE_V7 = {
    "base_args": {
        "steps": 600000,
        "batch_size": 16,
        "log_every": 10000,
        "save_every": 40000,
        "scaffolding": False,
        "seed": 42,
        "hidden": 100,
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,
        "tasks": POLI_TASKS,
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
        "foraging": "mvt",
        "forage_eps": 0.0,

        "rule_encoding": "lowrank",
        "rule_dim_low": 4,
        "rule_dim_out": 100,
    },
    "grid": [
        # V7 Grid — extends V6 with new levels per parameter:
        #   temp:     +0.1              (was 4, now 5)
        #   l1:       +5e-4             (was 3, now 4)
        #   dist_wt:  +5e-4             (was 2, now 3)
        #   beta:     +0.01             (was 3, now 4) — reaches infant tau_ratio range
        #   alpha:    +0.2              (was 3, now 4) — cognitive regression wants higher
        #   travel:   [50, 500]         (kept)
        #   eps:      [0.0, -0.01]      (kept)
        #
        # Total: L1 (4*4*4*5*2*2=1280) + Dist (3*4*4*5*2*2=960) = 2240
        # Existing V6 runs (720) auto-skipped by suffix matching → ~1520 new runs

        # 1. L1 Only (4 reg values × 4 beta × 4 alpha × 5 temp × 2 travel × 2 eps = 1280)
        {
            "l1_weight": [1e-6, 1e-5, 1e-4, 5e-4],
            "distance_penalty": [False],
            "forage_beta_global": [0.0001, 0.001, 0.003, 0.01],
            "forage_alpha_local": [0.01, 0.03, 0.1, 0.2],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1],
            "forage_temperature": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "forage_eps": [0.0, -0.01]
        },

        # 2. Distance Only (3 reg values × 4 beta × 4 alpha × 5 temp × 2 travel × 2 eps = 960)
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": [1e-5, 1e-4, 5e-4],
            "forage_beta_global": [0.0001, 0.001, 0.003, 0.01],
            "forage_alpha_local": [0.01, 0.03, 0.1, 0.2],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1],
            "forage_temperature": [0.0001, 0.001, 0.01, 0.1, 1.0],
            "forage_eps": [0.0, -0.01]
        },
    ]
}

SWEEP_CONFIG_FORAGE_V8 = {
    "base_args": {
        "steps": 600000,
        "batch_size": 16,
        "log_every": 10000,
        "save_every": 40000,
        "scaffolding": False,
        "seed": 42,
        "hidden": 100,
        "activation": "relu",
        "w_rec_init": "randortho",
        "w_rec_noise": 0.0,
        "sigma_x": 0.01,
        "grad_clip_mode": "value",
        "use_separate_input": False,
        "tasks": POLI_TASKS,
        "distance_path": "brain/assets/euclidean_distances_schaefer100.npy",
        "distance_power": 2.0,
        "foraging": "mvt",
        "forage_eps": 0.0,

        "rule_encoding": "lowrank",
        "rule_dim_low": 4,
        "rule_dim_out": 100,
    },
    "grid": [
        # V8 Grid — extends V7 with intermediate temperatures:
        #   temp:     +0.003, +0.005   (was 5, now 7) — locates MVT phase transition
        #   All other params: identical to V7
        #
        # Total: L1 (4*4*4*7*2*2=1792) + Dist (3*4*4*7*2*2=1344) = 3136
        # Existing V7 runs (2240) auto-skipped by suffix matching → 896 new runs

        # 1. L1 Only (4 reg × 4 beta × 4 alpha × 7 temp × 2 travel × 2 eps = 1792)
        {
            "l1_weight": [1e-6, 1e-5, 1e-4, 5e-4],
            "distance_penalty": [False],
            "forage_beta_global": [0.0001, 0.001, 0.003, 0.01],
            "forage_alpha_local": [0.01, 0.03, 0.1, 0.2],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1],
            "forage_temperature": [0.0001, 0.001, 0.003, 0.005, 0.01, 0.1, 1.0],
            "forage_eps": [0.0, -0.01]
        },

        # 2. Distance Only (3 reg × 4 beta × 4 alpha × 7 temp × 2 travel × 2 eps = 1344)
        {
            "l1_weight": [0.0],
            "distance_penalty": [True],
            "distance_weight": [1e-5, 1e-4, 5e-4],
            "forage_beta_global": [0.0001, 0.001, 0.003, 0.01],
            "forage_alpha_local": [0.01, 0.03, 0.1, 0.2],
            "forage_travel_steps": [50, 500],
            "forage_min_block_steps": [1],
            "forage_temperature": [0.0001, 0.001, 0.003, 0.005, 0.01, 0.1, 1.0],
            "forage_eps": [0.0, -0.01]
        },
    ]
}

CONFIGS = {
    "forage": SWEEP_CONFIG_FORAGE,
    "forage_v2": SWEEP_CONFIG_FORAGE_V2,
    "forage_v3": SWEEP_CONFIG_FORAGE_V3,
    "forage_v4": SWEEP_CONFIG_FORAGE_V4,
    "forage_v5": SWEEP_CONFIG_FORAGE_V5,
    "forage_v6": SWEEP_CONFIG_FORAGE_V6,
    "forage_v7": SWEEP_CONFIG_FORAGE_V7,
    "forage_v8": SWEEP_CONFIG_FORAGE_V8
}

# =============================================================================
# EXECUTION LOGIC
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run parameter sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--log-root", type=str, default="logs/sweep_forage", help="Root directory for sweep logs")
    parser.add_argument("--sweep-id", type=str, default=None, help="Optional name for the sweep folder")
    parser.add_argument("--config", type=str, default="forage", choices=list(CONFIGS.keys()), help="Choose sweep configuration")
    args = parser.parse_args()
    if args.log_root == "logs/sweep_forage":
         args.log_root = str(get_logs_dir() / "sweep_forage")
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
    raw_configs = []
    
    for sweep_set in sweep_sets:
        keys = list(sweep_set.keys())
        values = list(sweep_set.values())
        combinations = list(itertools.product(*values))
        
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            current = base_conf.copy()
            current.update(param_dict)
            raw_configs.append(current)

    # SORTING LOGIC:
    # Ensure Travel=50 runs come first, then Travel=500
    # We sort by 'forage_travel_steps'
    print("Sorting configurations by 'forage_travel_steps'...")
    raw_configs.sort(key=lambda x: x.get('forage_travel_steps', 0))

    # Assign Run Names in Order
    configs = []
    for i, conf in enumerate(raw_configs):
        # Neural naming logic
        parts = []
        if conf.get("distance_penalty"):
            parts.append(f"dist-{conf['distance_weight']:.0e}")
        elif conf.get("l1_weight", 0) > 0:
            parts.append(f"l1-{conf['l1_weight']:.0e}")
        else:
            parts.append("base")
            
        parts.append(f"beta{conf['forage_beta_global']}")
        parts.append(f"alpha{conf['forage_alpha_local']}")
        parts.append(f"trav{conf['forage_travel_steps']}")
        parts.append(f"blk{conf['forage_min_block_steps']}")
        parts.append(f"temp{conf['forage_temperature']}")
        
        # Add Epsilon if != 0 (Backward compat for runs where it was 0)
        if abs(conf.get("forage_eps", 0.0)) > 1e-9:
            parts.append(f"eps{conf['forage_eps']}")
        
        suffix = "_".join(parts)
        run_name = f"run_{i:03d}_{suffix}"
        
        conf["_run_name"] = run_name
        configs.append(conf)

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

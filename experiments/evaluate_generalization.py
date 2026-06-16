#!/usr/bin/env python3
"""
Script to evaluate a trained model on held-out tasks.

Usage:
    python experiments/evaluate_generalization.py --model_dir logs/run_xyz --tasks yang19.go-v0 yang19.dm1-v0 --steps 200

Outputs `eval_generalization.json` in the model directory.
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

# Ensure we can import from root
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from curiosity.data import make_dataset
from curiosity.metrics import evaluate_task
import analysis.plotting.utils as autils

# Global manual mapping for generalization tests
# Map requested task name (e.g. 'yang19.antigo-v0') to actual NeuroGym ID ('yang19.anti-v0')
TASK_ALIASES = {
    "yang19.antigo-v0": "yang19.anti-v0",  # Fix user typo
    "yang19.ctxdm1-v0": "yang19.ctxdm1-v0", # Correct
    "yang19.dm1-v0": "yang19.dm1-v0",       # Correct
    "yang19.go-v0": "yang19.go-v0",     # Correct
    "poli.dlyctxdm1": "poli.ctxdlydm1", # Correct name for Poli task (Registered as poli.ctxdlydm1)
}

# Map Poli/New tasks to the TRAINING TASK whose rule we should use.
# If a Poli task is a direct equivalent of a Yang task, we use the Yang task's rule index.
RULE_MAPPING = {
    # Poli Equivalent -> Yang Training Task
    "poli.go": "yang19.go-v0",
    "poli.antigo": "yang19.anti-v0",
    "poli.dm1": "yang19.dm1-v0",
    "poli.ctxdm1": "yang19.ctxdm1-v0",
    
    # Poli Compositional Tasks -> Which rule to use?
    "poli.antidm1": "yang19.anti-v0", 
    "poli.ctxgo": "yang19.go-v0",     
    "poli.dlyantigo": "yang19.anti-v0", 
    "poli.dlyctxdm1": "yang19.ctxdm1-v0", # Mapped to Context DM
    "poli.ctxdlydm1": "yang19.ctxdm1-v0"  # Alias target
}

def resolve_task_name(name: str) -> str:
    """Resolve aliases."""
    return TASK_ALIASES.get(name, name)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generalization on held-out tasks.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--tasks", nargs='+', default=[], help="List of tasks to evaluate on")
    parser.add_argument("--steps", type=int, default=200, help="Number of trials/steps per task")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint filename to load (e.g. model_last.pt)")
    parser.add_argument("--train-mode", action="store_true", help="Keep model in training mode (with noise) during eval")
    parser.add_argument("--sigma_x", type=float, default=0.0, help="Input noise level")
    return parser.parse_args()

def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
        
    device = torch.device(args.device)
    print(f"Loading model from {model_dir}... (Train Mode: {args.train_mode})")
    
    try:
        meta = autils.load_meta(model_dir)
        # We load the final requested state (or we could expose checkpoint selection)
        # For generalization, usually final model is desired.
        model = autils.build_model_from_meta(meta, device=device)
        
        # Try loading state dict
        if args.ckpt:
            ckpt_path = model_dir / args.ckpt
            print(f"Loading specific checkpoint: {ckpt_path}")
            obj = torch.load(ckpt_path, map_location=device)
            if isinstance(obj, dict) and "state_dict" in obj:
                state = obj["state_dict"]
            else:
                state = obj
        else:
            state = autils.load_state_dict_from_run(model_dir, map_location=device)
            
        autils.load_state_into_model(model, state)
        model.eval()
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    results: Dict[str, float] = {}
    
    print(f"Evaluating on tasks: {args.tasks}")
    
    for task_name in args.tasks:
        print(f"  Task: {task_name} ... ", end="", flush=True)
        print(f"  Task: {task_name} ... ", end="", flush=True)
        try:
            # 1. Resolve Alias
            actual_task_name = resolve_task_name(task_name)
            if actual_task_name != task_name:
                print(f"(alias for {actual_task_name}) ", end="")
            
            # Create dataset/env
            # We assume standard dt/batch_size from meta if available, or default
            dt = float(meta.get("cfg", {}).get("dt", 20.0))
            
            # Determine Rule Index
            n_rules_model = model.hp.n_rule
            train_tasks = meta.get("cfg", {}).get("tasks", [])
            
            # Logic:
            # 1. Is 'task_name' directly in train_tasks?
            # 2. Is there a mapping for 'task_name' to a train_task?
            # 3. Fallback to 0.
            
            target_rule_task = task_name 
            if task_name in RULE_MAPPING:
                target_rule_task = RULE_MAPPING[task_name]
                # If mapped, verify the mapped target is in training
            
            if target_rule_task in train_tasks:
                rule_idx = train_tasks.index(target_rule_task)
            elif actual_task_name in train_tasks:
                 # Check the resolved name too (e.g. yang19.anti-v0)
                 rule_idx = train_tasks.index(actual_task_name)
            else:
                # Fallback
                print(f"Warning: Task {task_name} (mapped: {target_rule_task}) not in training set. Using rule_idx=0.")
                rule_idx = 0
            
            # Create the dataset using the ACTUAL neurogym name (resolved alias)
            # n_rules SHOULD depend on the trained model configuration
            n_rules_train = len(train_tasks) if train_tasks else n_rules_model # fallback
            if n_rules_train == 0: n_rules_train = 1
            
            ds = make_dataset(actual_task_name, dt=dt, batch_size=1, rule_idx=rule_idx, n_rules=n_rules_train)
            
            # evaluate_task reads from env.ob (raw) so it needs to append the rule itself.
            acc = evaluate_task(model, ds.env, device=device, ntrial=args.steps,
                                rule_idx=rule_idx, n_rules=n_rules_train,
                                train_mode=args.train_mode,
                                sigma_x=args.sigma_x)
            results[task_name] = acc
            print(f"Acc: {acc:.4f}")
            
        except Exception as e:
            import traceback
            with open("eval_errors.log", "a") as errf:
                errf.write(f"Task: {task_name}\n")
                traceback.print_exc(file=errf)
                errf.write("\n" + "="*30 + "\n")
            print(f"Error: {e}")
            results[task_name] = -1.0

    # Save results
    out_path = model_dir / "eval_generalization.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results to {out_path}")
    
    # Plot results
    if results:
        import matplotlib.pyplot as plt
        
        tasks = list(results.keys())
        accs = list(results.values())
        
        plt.figure(figsize=(10, 6))
        # Use short names for x-axis if possible
        short_names = [t.split('.')[-1] for t in tasks]
        
        bars = plt.bar(short_names, accs, color='skyblue')
        plt.ylim(0, 1.05)
        plt.ylabel("Accuracy")
        plt.title("Zero-Shot Generalization Performance")
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save to plots subdir
        plots_dir = model_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_path = plots_dir / "eval_generalization.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()

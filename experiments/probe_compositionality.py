#!/usr/bin/env python3
"""
Probe Compositionality Script
Tests if a frozen RNN can solve a held-out task by *only* learning a new input rule vector.
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import math

# Ensure we can import from root
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from curiosity.data import make_dataset
import analysis.plotting.utils as autils
from curiosity.metrics import evaluate_task, ce_yang_cmask
from curiosity.models.yang19_khona import Yang19KhonaModel

def probe_task(model, task_name, device, steps=2000, batch_size=32, lr=1e-2, 
               new_rule_idx=None):
    """
    Runs the few-shot probe.
    params:
      model: The loaded model (will be modified in-place, assumed frozen outside)
      task_name: Target task (e.g. 'poli.ctxantigo')
      steps: Max optimization steps (batches)
    """
    print(f"\n--- Probing {task_name} ---")
    
    # 1. Dataset
    # make_dataset needs n_rules to match the Model's NEW size
    n_rules = model.hp.n_rule 
    dt = model.hp.dt
    print(f"DEBUG: Dataset n_rules={n_rules}, rule_idx={new_rule_idx}")
    # Use 'episode' - safer for logical composition
    ds = make_dataset(task_name, dt=dt, batch_size=batch_size, 
                      rule_idx=new_rule_idx, n_rules=n_rules, dataset_mode='episode')
    
    # Check shape
    ob_sample, gt_sample = ds()
    print(f"DEBUG: Sample Batch Shape: {ob_sample.shape}")
    
    # 2. Setup Optimizer
    # We only optimize the parameters that require_grad
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        print("Error: No active parameters found!")
        return {}
        
    optimizer = optim.Adam(params, lr=lr)
    
    # 3. Training Loop
    log_freq = 100
    history = []
    
    solved_at = None
    
    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        
        # Get Batch
        ob, gt = ds()
        x = torch.from_numpy(ob).float().to(device) # (T, B, D)
        y = torch.from_numpy(gt).long().to(device)  # (T, B)
        
        # Add noise? Using Yang defaults
        x += torch.randn_like(x) * model.hp.sigma_rec * (2.0/0.2)**0.5
        
        # Forward
        outs = model(x)
        logits = outs['ring_logits'] # (T, B, C)
        
        # Loss
        loss = ce_yang_cmask(logits, y)
        loss.backward()
        
        # Clip grad?
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        # Evaluate periodically
        if (step+1) % log_freq == 0:
            model.eval()
            # ds.env is already a RuleAugmentedEnv that appends the rule vector to ob;
            # do NOT pass rule_idx/n_rules here or they get appended a second time.
            acc = evaluate_task(model, ds.env, device=device, ntrial=50)
                                
            print(f"Index {step+1}: Loss={loss.item():.4f}, Acc={acc:.2f}")
            history.append({'step': step+1, 'loss': loss.item(), 'acc': acc})
            
            if acc > 0.9 and solved_at is None:
                solved_at = step + 1
                print(f"SOLVED at step {solved_at}!")
            
            # Early stop if perfect
            if acc >= 0.99:
                print(f"Perfect accuracy reached. Stopping early at {step+1}.")
                break
                
    final_acc = history[-1]['acc'] if history else 0.0
    return {
        'task': task_name,
        'solved_at': solved_at,
        'final_acc': final_acc,
        'history': history
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--tasks", nargs='+', default=['poli.ctxantigo'])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-readout", action='store_true', help="If set, allow training W_out.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    
    # 1. Load Model
    meta = autils.load_meta(model_dir)
    # Rebuild model structure
    model_orig = autils.build_model_from_meta(meta, device=device)
    state = autils.load_state_dict_from_run(model_dir, map_location=device)
    autils.load_state_into_model(model_orig, state)
    
    print(f"Loaded model. Rules: {model_orig.hp.n_rule}")
    
    # 2. Surgery
    old_n_rule = model_orig.hp.n_rule
    new_n_rule = old_n_rule + 1
    hidden_size = model_orig.hp.n_rnn
    
    if model_orig.hp.use_separate_input:
        # --- Separate Input Architecture ---
        print("Architecture: Separate Input (Sensory vs Rule)")
        old_rule_in = model_orig.rule_in # nn.Linear(old_n_rule, n_rnn, bias=False)
        
        # New Weight (Out, In) -> (n_rnn, new_n_rule)
        new_weight = torch.zeros(hidden_size, new_n_rule).to(device)
        new_weight[:, :old_n_rule] = old_rule_in.weight.data
        nn.init.normal_(new_weight[:, -1:], std=0.01)
        
        # Replace Layer
        model_orig.rule_in = nn.Linear(new_n_rule, hidden_size, bias=False).to(device)
        model_orig.rule_in.weight.data = new_weight
        
        # Freeze Logic
        for name, p in model_orig.named_parameters(): p.requires_grad = False
        model_orig.rule_in.weight.requires_grad = True
        
        def reset_fn():
            with torch.no_grad():
                # Reset last column of rule_in
                nn.init.normal_(model_orig.rule_in.weight.data[:, -1:], std=0.1)
        
    else:
        W_old = model_orig.cell.W.data # (In+Rec, Hid)
        n_w_in, n_hid = W_old.shape
        actual_input_size = n_w_in - n_hid
        
        print(f"DEBUG: W_old shape {W_old.shape}. Implied input size: {actual_input_size}")
        if model_orig.hp.n_input != actual_input_size:
            print(f"WARNING: HP n_input ({model_orig.hp.n_input}) != Weights ({actual_input_size}). Syncing...")
            model_orig.hp.n_input = actual_input_size
            
        # Always sync n_rule based on input
        # Assuming 33 sensory dims (fixed for Poli/Yang19)
        implied_rules = model_orig.hp.n_input - 33
        if implied_rules > 0 and model_orig.hp.n_rule != implied_rules:
             print(f"WARNING: Updating HP n_rule from {model_orig.hp.n_rule} to {implied_rules} (derived from input {model_orig.hp.n_input})")
             model_orig.hp.n_rule = implied_rules
        
        old_n_rule = model_orig.hp.n_rule
        old_n_input = model_orig.hp.n_input
        new_n_rule = old_n_rule + 1
        
        print(f"DEBUG: Surgery - Old Rules: {old_n_rule}, New Rules: {new_n_rule}")
        
        # Split
        W_in_old = W_old[:old_n_input, :]
        W_rec_old = W_old[old_n_input:, :]
        
        # Expand Input
        # New input has 1 extra row at the end
        new_row = torch.zeros(1, hidden_size).to(device)
        nn.init.normal_(new_row, std=0.01) # Small random init
        
        W_in_new = torch.cat([W_in_old, new_row], dim=0) # (OldIn+1, Hid)
        
        # Re-fuse
        W_total_new = torch.cat([W_in_new, W_rec_old], dim=0)
        
        # Replace Parameter
        model_orig.cell.W = nn.Parameter(W_total_new)
        
        # Freeze Logic
        for name, p in model_orig.named_parameters(): p.requires_grad = False
        model_orig.cell.W.requires_grad = True
        
        # Hook: Mask everything except the new input row
        # The new row index is old_n_input (0-indexed)
        target_row_idx = old_n_input 
        
        def zero_old_grads_hook(grad):
            # grad shape (NewIn+Rec, Hid)
            mask = torch.zeros_like(grad)
            mask[target_row_idx, :] = 1.0
            return grad * mask
            
        model_orig.cell.W.register_hook(zero_old_grads_hook)
        
        def reset_fn():
            with torch.no_grad():
                # Reset the target row
                nn.init.normal_(model_orig.cell.W.data[target_row_idx:target_row_idx+1, :], std=0.1)

    # Common Updates
    model_orig.hp.n_rule = new_n_rule
    model_orig.hp.n_input += 1
    
    # Readout Logic
    if args.train_readout:
        print("Mode: Unfrozen Readout (Information Probe)")
        model_orig.w_out.weight.requires_grad = True
        model_orig.w_out.bias.requires_grad = True
    else:
        print("Mode: Frozen Readout (Alignment Probe)")
        
    # Verify Active Params
    print("Active Parameters:")
    for name, p in model_orig.named_parameters():
        if p.requires_grad:
            print(f"  {name} {p.shape}")
            
    # 4. Run Probes
    results = {}
    for task in args.tasks:
        # Reset the new rule vector for each task to a fresh random state
        reset_fn()
             
        res = probe_task(model_orig, task, device, steps=args.steps, 
                         new_rule_idx=new_n_rule-1)
        results[task] = res
        
    # Save
    out_path = model_dir / "probe_compositionality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to compare multiple experiments from a sweep directory.
Now includes:
- Auto-running generalization evaluation if missing.
- Detailed plots with individual task performance (dots) and means.
- Grouping by Regularization Type (L1, Proximal, Distance).
"""

import argparse
import sys
import json
import getpass
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import subprocess
import re
from scipy.stats import entropy
from pathlib import Path
from typing import List, Dict, Any

import logging
# Suppress findfont warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Ensure correct path for imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import analysis.plotting.utils as autils
import torch
from experiments.run_sweep_slurm import submit_job_daemon
from analysis.utils.paths import get_logs_dir, resolve_sweep_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Compare experiments.")
    parser.add_argument("--sweep_dirs", type=str, nargs='+', required=True, help="List of sweep directories to compare")
    parser.add_argument("--window", type=int, default=100, help="Smoothing window for final accuracy")
    parser.add_argument("--device", type=str, default="cpu", help="Device for model loading")
    parser.add_argument("--mode", type=str, default="winners", choices=["winners", "all", "none", "targeted"], 
                        help="Analysis mode: 'winners' (only best models), 'all' (all models), 'none' (agg only), 'targeted' (specific runs)")
    parser.add_argument("--compute-gen", action="store_true", help="Compute generalization (probe/eval) if missing")
    parser.add_argument("--target-runs", type=str, nargs='+', default=[], help="List of run ID strings (e.g. '91' '42') for targeted analysis")
    parser.add_argument("--execution-mode", type=str, default="local", choices=["local", "cluster"], help="Run analysis locally or submit to cluster")
    parser.add_argument("--gen-method", type=str, default="probe", choices=["probe", "zero_shot"],
                        help="Generalization method: 'probe' (few-shot optimization) or 'zero_shot' (static eval)")
    parser.add_argument("--probe-steps", type=int, default=2000, help="Optimization steps for probe method")
    parser.add_argument("--gen-tasks", type=str, nargs="+", default=None, help="Specific tasks to evaluate (overrides set choice)")
    parser.add_argument("--gen-set", type=str, default="quick", choices=["quick", "full"], help="Pre-defined test set: 'quick' (6 tasks) or 'full' (34 tasks)")
    parser.add_argument("--no-auto-analysis", action="store_true", help="Do NOT auto-run run_analysis.py on best models")
    parser.add_argument("--max-plot-steps", type=int, default=None, help="Limit x-axis of plots to N steps (e.g. 1000)")
    
    args = parser.parse_args()
    return args

def get_full_test_set() -> List[str]:
    """
    Return the FULL set (34 Tasks): 20 Training + 14 Held-out.
    """
    return [
        # --- TRAINING CONTROLS (20 Tasks) ---
        # Reach
        'poli.go', 'poli.rtgo', 'poli.dlygo', 'poli.antigo', 'poli.ctxgo',
        'poli.dlyantigo', 'poli.dlyctxgo', 'poli.antictxgo', 'poli.dlyantictxgo', 'poli.rtantictxgo',
        # Decision
        'poli.dm1', 'poli.dm2', 'poli.dlydm1', 'poli.multidm', 'poli.antidm1', 'poli.antidlydm1',
        # Match
        'poli.dlyms', 'poli.dlynms', 'poli.catdlyms', 'poli.catdlynms',

        # --- HELD-OUT TARGETS (14 Tasks) ---
        # Context -> Decision
        'poli.ctxdm1', 'poli.ctxdm2',
        'poli.ctxdlydm1', 'poli.ctxdlydm2',
        'poli.antictxdm1', 'poli.antictxdm2',
        'poli.antictxdlydm1', 'poli.antictxdlydm2',
        # Anti -> Match
        'poli.antidlyms', 'poli.antidlynms',
        # Context + Anti -> Match
        'poli.antictxdlyms', 'poli.antictxcatdlyms'
    ]

def get_quick_test_set() -> List[str]:
    """
    Return the QUICK set (6 Tasks): 3 Targets + 3 Matched Controls.
    Used for rapid probing.
    """
    return [
        # Controls (Training)
        'poli.antidm1',     # 2 Ops (Matches ctxdm1)
        'poli.dlyantigo',   # 3 Ops (Matches antidlyms)
        'poli.dlyantictxgo',# 4 Ops (Matches antictxdlyms)
        
        # Targets (Held-out)
        'poli.ctxdm1',      # 2 Ops
        'poli.antidlyms',   # 3 Ops
        'poli.antictxdlyms' # 4 Ops
    ]

def calculate_foraging_metrics(run_dir: Path) -> Dict[str, float]:
    """
    Parse progress.csv to compute:
    - Switch Rate: (Unique Blocks / Total Steps) * 1000
    - Behavioral Entropy: Entropy of task distribution
    """
    csv_path = run_dir / "run_curriculum.csv"
    if not csv_path.exists():
        csv_path = run_dir / "progress.csv"
        if not csv_path.exists():
            return {'switch_rate': 0.0, 'entropy': 0.0}
    
    try:
        # Load only necessary columns to save memory
        df = pd.read_csv(csv_path, usecols=['mode', 'task'])
        
        # Filter for Training phase (ignore travel steps for task distribution? 
        # Actually travel steps ARE the "exploration cost".
        # But 'task' column during travel is the ORIGIN task usually.
        # Let's look at the sequence of tasks VISITED.
        
        # 1. Identify Blocks
        # A block change is when 'task' changes.
        # But during travel, task might remain same or be 'TRAVEL'.
        # Let's just count transitions in the 'task' column.
        
        # Remove consecutive duplicates
        tasks = df['task'].values
        if len(tasks) == 0: return {'switch_rate': 0.0, 'entropy': 0.0}
        
        # Transitions: task[i] != task[i-1]
        # We need to handle 'TRAVEL' mode? 
        # In loops.py earlier, task IS reported during travel.
        # Let's count "Task Switches".
        n_steps = len(tasks)
        transitions = np.sum(tasks[1:] != tasks[:-1])
        switch_rate = (transitions / n_steps) * 1000
        
        # 2. Entropy (Diversity of Tasks)
        # Fraction of time spent in each task
        counts = df['task'].value_counts(normalize=True)
        ent = entropy(counts)
        
        return {
            'switch_rate': switch_rate,
            'entropy': ent,
            'n_steps': n_steps
        }
    except Exception as e:
        print(f"[{run_dir.name}] Error reading progress.csv: {e}")
        return {'switch_rate': 0.0, 'entropy': 0.0}

def run_evaluation_if_needed(run_dir: Path, tasks: List[str], method: str = "probe", steps: int = 2000):
    """Run generalization evaluation if missing."""
    
    if method == "probe":
        gen_path = run_dir / "probe_compositionality.json"
        script_name = "experiments/probe_compositionality.py"
    else:
        gen_path = run_dir / "eval_generalization.json"
        script_name = "experiments/evaluate_generalization.py"
    
    should_run = False
    if not gen_path.exists():
        should_run = True
    else:
        try:
            with open(gen_path, "r") as f:
                data = json.load(f)
            # Check coverage
            if not all(t in data for t in tasks):
                print(f"[{run_dir.name}] Existing {gen_path.name} missing required tasks. Re-running...")
                should_run = True
        except:
            should_run = True

    if should_run and tasks:
        print(f"[{run_dir.name}] Running {method} eval on {len(tasks)} tasks...")
        
        cmd = [sys.executable, str(root_dir / script_name)]
        
        if method == "probe":
            cmd.extend([
                "--model_dir", str(run_dir),
                "--tasks"
            ])
            cmd.extend(tasks)
            cmd.extend(["--steps", str(steps)])
        else:
            # Zero-shot arguments
            cmd.extend([
                "--model_dir", str(run_dir),
                "--ckpt", "model_last.pt",
                "--tasks"
            ])
            cmd.extend(tasks)
        
        try:
            # allow stdout to flow to console so user sees progress
            subprocess.run(cmd, check=True)
            print(f"[{run_dir.name}] Evaluation complete.")
        except subprocess.CalledProcessError as e:
            print(f"[{run_dir.name}] Evaluation FAILED.")

def calculate_per_task_speed(ema_dict, threshold=0.8):
    """Return a dict of task_name -> steps_to_reach_threshold."""
    speeds = {}
    for task, acc_series in ema_dict.items():
        # acc_series is a list or array
        arr = np.array(acc_series)
        # Smooth lightly?
        # Find first index >= threshold
        idxs = np.where(arr >= threshold)[0]
        if len(idxs) > 0:
            speeds[task] = int(idxs[0])
        else:
            speeds[task] = np.nan # Did not reach
    return speeds

def get_weight_evolution(run_dir: Path, device: str = "cpu") -> List[Dict[str, Any]]:
    """
    Extract total sum of |W_rec| across checkpoints.
    Returns list of dicts: [{'step': s, 'l1_sum': val}, ...]
    """
    try:
        from analysis.plotting.utils import (
            get_Wrec,
            list_checkpoint_steps_and_paths,
            build_model_from_meta,
            load_state_into_model,
            load_meta
        )
        
        meta = load_meta(run_dir)
        ckpts = list_checkpoint_steps_and_paths(run_dir)
        
        if not ckpts:
            return []
            
        model = build_model_from_meta(meta, device=torch.device(device))
        
        results = []
        for step, path in ckpts:
            try:
                obj = torch.load(path, map_location=device)
                if isinstance(obj, dict) and "state_dict" in obj:
                    state = obj["state_dict"]
                else:
                    state = obj
                
                load_state_into_model(model, state)
                l1 = get_Wrec(model).detach().abs().sum().item()
                results.append({'step': step, 'l1_sum': l1})
            except Exception as e:
                print(f"Error loading checkpoint {path}: {e}")
                continue
                
        return results
    except Exception as e:
        print(f"[{run_dir.name}] Failed to extract weights: {e}")
        return []

def load_run_data(run_dir: Path, smooth_window: int, heldout_tasks: List[str], device: str, run_source: str, compute_gen: bool = False, gen_method: str = "probe", probe_steps: int = 2000) -> Dict[str, Any]:
    """Extract metrics from a single run."""
    # print(f"[{run_dir.name}] Loading... compute_gen={compute_gen}, method={gen_method}")
    
    # 1. Config & Identify Type
    reg_type = "Unknown"
    weight = 0.0
    try:
        meta = autils.load_meta(run_dir)
        cfg = meta.get("cfg", {})
        l1 = float(cfg.get("l1_weight", 0.0))
        prox = float(cfg.get("prox_l1_weight", 0.0))
        dist = float(cfg.get("distance_weight", 0.0))
        dist = float(cfg.get("distance_weight", 0.0))
        dist_pen = cfg.get("distance_penalty", False)

        # NOTE: heldout_tasks is passed in via argument (from main), which defaults to get_default_test_tasks()
        # but can be overridden by --gen-tasks. We do NOT override it here.

        # Foraging Params
        # Try to get from Config first, fallback to regex on run_name
        beta = float(cfg.get("forage_beta_global", 0.003))
        travel = int(cfg.get("forage_travel_steps", 50))
        temp = float(cfg.get("forage_temperature", 0.0))
        
        # Regex Fallback if config is default but run name implies sweep
        # Run Name: run_001_l1-1e-05_dist-0.0001_beta0.003_trav50_temp0.05
        name = run_dir.name
        
        m_beta = re.search(r"beta([\d\.]+)", name)
        if m_beta: beta = float(m_beta.group(1))
            
        m_travel = re.search(r"trav(\d+)", name)
        if m_travel: travel = int(m_travel.group(1))
            
        m_temp = re.search(r"temp([\d\.]+)", name)
        if m_temp: temp = float(m_temp.group(1))

        min_block = int(cfg.get("forage_min_block_steps", 10))
        m_blk = re.search(r"blk(\d+)", name)
        if m_blk: min_block = int(m_blk.group(1))

        alpha = float(cfg.get("forage_alpha_local", 0.03))
        m_alpha = re.search(r"alpha([\d\.]+)", name)
        if m_alpha: alpha = float(m_alpha.group(1))
        
        # Determine Type and Main Weight
        if dist > 0:
            if l1 > 0:
                reg_type = "L1 + Distance"
                weight = dist # Or maybe a tuple? But for X-axis plotting we need a scalar. 
                              # Let's use Distance weight as primary X, since L1 is likely fixed (1e-4).
            else:
                reg_type = "Distance"
                weight = dist
        elif prox > 0:
            reg_type = "Proximal L1"
            weight = prox
        elif l1 > 0:
            reg_type = "L1"
            weight = l1
        else:
            reg_type = "Baseline"
            weight = 0.0 
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading {run_dir}: {e}")
        return None

    # Auto-Evaluate if needed
    if compute_gen:
        run_evaluation_if_needed(run_dir, heldout_tasks, method=gen_method, steps=probe_steps)

    # 4. Weight Evolution
    weight_history = get_weight_evolution(run_dir, device)

    # 5. Foraging Metrics (New!)
    f_metrics = calculate_foraging_metrics(run_dir)

    # 2. Training Performance (Per Task)
    rows = autils.load_curriculum_rows(run_dir)
    if not rows:
        return None
        
    _, ema_dict = autils.build_per_task_ema_dict(rows) 
    
    # Get final accuracies per task
    final_accs = {}
    for t, history in ema_dict.items():
        if history:
            final_accs[t] = np.mean(history[-smooth_window:]) if len(history) > smooth_window else np.mean(history)
            
    # Get speeds per task
    # Get speeds per task
    speeds = calculate_per_task_speed(ema_dict, threshold=0.8)

    # Calculate Mean Speed (Average steps to solve for SOLVED tasks)
    solved_speeds = [v for v in speeds.values() if not np.isnan(v)]
    mean_speed = np.mean(solved_speeds) if solved_speeds else np.nan

    # 3. Generalization (Per Task)
    gen_accs = {}
    # Always attempt to load if file exists
    if True:
        # Target Check depending on method
        if gen_method == "probe":
            gen_path = run_dir / "probe_compositionality.json"
        else:
            gen_path = run_dir / "eval_generalization.json"

        if gen_path.exists():
            try:
                with open(gen_path, "r") as f:
                    gen_data = json.load(f)
                    
                if gen_method == "probe":
                    # Parse nested structure: key -> {final_acc: X}
                    for t, vals in gen_data.items():
                        if isinstance(vals, dict) and "final_acc" in vals:
                            gen_accs[t] = vals["final_acc"]
                else:
                    # Parse flat structure: key -> acc
                    gen_accs = {k: v for k, v in gen_data.items() if v >= 0}
            except:
                pass
            
    return {
        'run_source': run_source,
        'run_name': run_dir.name,
        'reg_type': reg_type,
        'weight': weight,
        'l1_val': l1,
        'dist_val': dist,
        'final_accs': final_accs,
        'speeds': speeds,
        'gen_accs': gen_accs,
        'weight_history': weight_history,
        'step': weight_history[-1]['step'] if weight_history else 0,
        'l1_sum': weight_history[-1]['l1_sum'] if weight_history else 0,
        'type': reg_type, # Redundant with reg_type, but keeping for consistency with user's request
        'l1': l1,
        'prox': prox,
        'dist': dist,
        'beta': beta,
        'travel': travel,
        'temp': temp,
        'switch_rate': f_metrics['switch_rate'],
        'entropy': f_metrics['entropy'],
        'alpha': alpha,
        'min_block': min_block,
        'mean_speed': mean_speed
    }

def run_analysis_on_models(target_runs, out_dir, args):
    """
    Runs analysis/run_analysis.py on the target models.
    Mode:
    - local: Runs fully serially (subprocess check_call).
    - cluster: Submits 1 job per run to the cluster (using submit_job_daemon).
    """
    if args.no_auto_analysis:
        print("\n[Auto-Analysis] Skipped (--no-auto-analysis flag set)")
        return
        
    print("\n" + "="*60)
    print("AUTO-RUNNING ANALYSIS ON BEST MODELS")
    print("="*60)
    
    # Check if analysis script exists
    root_dir = Path(__file__).resolve().parent.parent
    analysis_script = root_dir / "analysis" / "run_analysis.py"
    
    if not analysis_script.exists():
        print(f"Analysis script not found at {analysis_script}")
        return
    
    print(f"Running analysis on {len(target_runs)} models...")
    
    # Pre-scan for cluster submission
    user = getpass.getuser()
    
    for run_path in target_runs:
        
        # Output directory: comparison_plots/run_XXX_analysis
        analysis_out = out_dir / f"{run_path.name}_analysis"
        analysis_out.mkdir(exist_ok=True, parents=True)
        
        # Check if probe file exists
        # Probe script writes to run_path, NOT analysis_out
        skip_submission = False
        if (run_path / "probe_compositionality.json").exists():
            print(f"[SKIP] Already probed: {run_path.name}")
            skip_submission = True

        print(f"\n[{run_path.name}] Processing logic...")

        # Construct Analysis Command Arguments (Script-relative)
        # Note: We pass raw string args for subprocess/slurm
        # Important: run_analysis.py expects --run_dir etc.
        # For cluster, the paths must be valid on the cluster. 
        # Assuming current paths (networked/Absolute) work on both or we rely on the user to use consistent mounts.
        
        cmd_args = [
            "--run_dir", str(run_path),
            "--save_dir", str(analysis_out),
            "--no-plot_foraging",
            "--no-plot_scaffolding"
        ]
        
        if args.max_plot_steps:
             cmd_args.extend(["--step_range", "0", str(args.max_plot_steps)])
        
        if args.execution_mode == 'local':
            if not skip_submission:
                print(f"   ► Running Local Analysis...")
                full_cmd = [sys.executable, str(analysis_script)] + cmd_args
                try:
                    subprocess.run(full_cmd, check=True, capture_output=True)
                    print(f"   ✓ Analysis saved to: {analysis_out}")
                except subprocess.CalledProcessError as e:
                    print(f"   ✗ Analysis failed: {e}")
                    if e.stderr:
                        err_msg = e.stderr.decode('utf-8', errors='ignore')[-500:]
                        print(f"   Error: {err_msg}")
            else:
                 print(f"   [Skipping Local Probe Execution as it exists]")

        elif args.execution_mode == 'cluster' and not skip_submission:
            print(f"   ► Submitting to Cluster...")
            # We use submit_job_daemon but in specific 'analysis' context
            # run_name = Job Name
            # cmd_str = Arguments string
            # args = Needs Slurm constraints (execution_mode is in args, but Slurm params are usually defaults)
            # CAUTION: args here is from compare_experiments, which might lack slurm-specific args (queue, time).
            # We should patch args with defaults if missing.
            
            # Patch Mock Args for Slurm
            class SlurmArgs:
                partition = "Main"
                time = "48:00:00" # Increase if probes are slow
                mem = "8G"
                cpus = 4
            
            slurm_args = SlurmArgs()
            
            # For Probe, we need specific arguments
            # args.gen_method should be "probe"
            if args.gen_method == "probe":
                target_script = "experiments/probe_compositionality.py"
                tasks_to_run = get_full_test_set() if args.gen_set == "full" else get_quick_test_set()
                cmd_str = f"--model_dir {run_path} --steps {args.probe_steps} --tasks {' '.join(tasks_to_run)}"
                job_prefix = "prb"
            else:
                # Fallback or Zero Shot
                target_script = "experiments/evaluate_generalization.py"
                tasks_to_run = get_full_test_set() if args.gen_set == "full" else get_quick_test_set()
                cmd_str = f"--model_dir {run_path} --ckpt model_last.pt --tasks {' '.join(tasks_to_run)}"
                job_prefix = "gen"

            # Job Name
            job_name = f"{job_prefix}_{run_path.name}"

            # Submit
            # We use a dummy 'base_dir' for logs because run_sweep_slurm expects one
            log_dir = out_dir / "_slurm_logs"
            log_dir.mkdir(exist_ok=True)
            
            success, msg = submit_job_daemon(
                run_name=job_name,
                cmd_str=cmd_str,
                args=slurm_args, # Use our mocked constraints
                base_dir=log_dir,
                qos="normal",
                node_line="", # No specific node constraints (Any) or we could scan
                python_script=target_script # TARGET SCRIPT
            )
            
            if success:
                print(f"     ✓ Submitted {job_name}: {msg}")
            else:
                print(f"     ✗ Submission Failed: {msg}")

        # --- ALWAYS Run Standard Analysis locally for plotting ---
        # Since we are processing winners, N is small.
        # This populates the analysis folder with standard plots.
        print(f"   ► Generating Standard Plots locally...")
        std_cmd = [
            sys.executable, str(analysis_script),
            "--run_dir", str(run_path),
            "--save_dir", str(analysis_out),
            "--no-plot_scaffolding" # Optional, based on pref
        ]
        # Basic plots only (train, weights, forage if applicable)
        try:
            subprocess.run(std_cmd, check=True, capture_output=True)
            print(f"     ✓ Standard plots saved.")
        except subprocess.CalledProcessError as e:
            print(f"     ✗ Standard plotting failed: {e}")


def plot_probe_learning_curves(model_dir: Path, output_dir: Path, gen_method: str = "probe"):
    """
    Plot learning curves from probe compositionality data.
    Shows how accuracy evolves over optimization steps for each generalization task.
    Style: All tasks in single plot, identified by color (like training_per_task.png).
    """
    if gen_method != "probe":
        return  # Only applicable for probe method
        
    probe_path = model_dir / "probe_compositionality.json"
    if not probe_path.exists():
        print(f"   No probe data found for {model_dir.name}")
        return
    
    try:
        with open(probe_path, "r") as f:
            probe_data = json.load(f)
    except:
        print(f"   Error loading probe data for {model_dir.name}")
        return
    
    # Extract learning curves
    fig = plt.figure(figsize=(10, 6))
    
    for task_name, task_data in probe_data.items():
        if not isinstance(task_data, dict) or 'history' not in task_data:
            continue
            
        history = task_data['history']
        if not history:
            continue
        
        steps = [h['step'] for h in history]
        accs = [h['acc'] for h in history]
        
        plt.plot(steps, accs, label=task_name, linewidth=2, alpha=0.8)
    
    plt.xlabel("Optimization Step")
    plt.ylabel("Accuracy")
    plt.title(f"Probe Learning Curves: {model_dir.name}")
    plt.legend(loc='best', fontsize=9)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "probe_learning_curves.png"
    plt.savefig(output_path)
    plt.close()
    print(f"   ✓ Saved probe learning curves to: {output_path}")


def _run_targeted(args, run_dirs, out_dir):
    """Fast path for --mode targeted: match IDs, submit probe jobs, plot LC. No data loading."""
    import re as _re

    if not args.target_runs:
        print("Error: --mode targeted requires --target-runs list.")
        sys.exit(1)

    run_name_to_path = {r_path.name: r_path for r_path, _ in run_dirs}

    print(f"Filtering for TARGETED runs containing: {args.target_runs}")
    target_matches = []
    for r_name in run_name_to_path.keys():
        m = _re.search(r"^run_(\d+)_", r_name)
        run_id = int(m.group(1)) if m else None
        for t in args.target_runs:
            if t.isdigit():
                if run_id is not None and int(t) == run_id:
                    target_matches.append(r_name)
                    break
            else:
                if t in r_name:
                    target_matches.append(r_name)
                    break

    target_run_ids = list(set(target_matches))
    print(f"Found {len(target_run_ids)} targeted runs: {target_run_ids}")

    # Probe learning curve plots (only if data already exists)
    if args.gen_method == "probe":
        print("\n[+] Generating Probe Learning Curve Plots for Selected Models...")
        for r_name in target_run_ids:
            r_path = run_name_to_path[r_name]
            analysis_out = out_dir / f"{r_name}_analysis"
            analysis_out.mkdir(exist_ok=True, parents=True)
            plot_probe_learning_curves(r_path, analysis_out, args.gen_method)

    # Submit / run probe jobs
    target_paths = [run_name_to_path[r] for r in target_run_ids if r in run_name_to_path]
    run_analysis_on_models(target_paths, out_dir, args)


def main():
    print("Starting analysis script...", flush=True)
    args = parse_args()
    sweep_dirs = []

    for d_str in args.sweep_dirs:
         try:
             d = resolve_sweep_dir(d_str)
         except Exception as e:
             print(f"Error resolving directory: {d_str} - {e}", flush=True)
             sys.exit(1)
             
         if not d.exists():
             print(f"Error: Directory not found: {d_str} (resolved to {d})", flush=True)
             sys.exit(1)
         
         sweep_dirs.append(d)

    # Identify tasks
    if args.gen_tasks:
        heldout = args.gen_tasks
        print(f"Using Custom Task List: {heldout}")
    else:
        if args.gen_set == "full":
            heldout = get_full_test_set()
            print(f"Using FULL Test Set (34 Tasks)")
        else:
            heldout = get_quick_test_set()
            print(f"Using QUICK Test Set (6 Tasks)")

    run_dirs = []
    # Collect runs from all sweep dirs, tagging them with source
    for sd in sweep_dirs:
        found = [d for d in sd.iterdir() if d.is_dir() and (d / "model_meta.json").exists()]
        print(f"Found {len(found)} runs in {sd.name}.", flush=True)
        # Store as tuple (path, source_name)
        for d in found:
            run_dirs.append((d, sd.name))
            
    print(f"Total runs to analyze: {len(run_dirs)}", flush=True)
    
    # Collect flattened data for plotting
    perf_records = []
    speed_records = []
    gen_records = []
    weight_records = []
    agg_records = []




    for d, source in run_dirs:
        # In targeted mode we only need paths (built from run_dirs directly below).
        # Skip all expensive I/O — model_meta, training logs, weight history, foraging metrics.
        if args.mode == "targeted":
            continue

        print(f".. [{source}] {d.name}")

        # If running on cluster, disable local generalization probing during the scan
        # We only want to load basic metrics to find the winners
        # Only compute GEN during scan if mode is "all" and compute-gen is requested
        should_compute_gen = (args.mode == "all" and args.compute_gen)
        if args.execution_mode == "cluster":
            should_compute_gen = False

        data = load_run_data(d, args.window, heldout, args.device, run_source=source,
                             compute_gen=should_compute_gen, gen_method=args.gen_method, probe_steps=args.probe_steps)
        if not data:
            continue
            
        r_type = data['reg_type']
        w = data['weight']
        l1_v = data.get('l1_val', 0.0)
        dist_v = data.get('dist_val', 0.0)
        run = data['run_name']
        source = data['run_source']
        
        # Performance
        for t, val in data['final_accs'].items():
            perf_records.append({
                'Run': run, 'Source': source, 'Type': r_type, 'Weight': w, 'Task': t, 'Value': val, 
                'L1': l1_v, 'Dist': dist_v, 'Beta': data['beta'], 'Travel': data['travel'], 'Temp': data['temp'], 'Alpha': data['alpha'], 'MinBlock': data['min_block']
            })
            
        # Speed
        for t, val in data['speeds'].items():
            if not np.isnan(val):
                 speed_records.append({
                     'Run': run, 'Source': source, 'Type': r_type, 'Weight': w, 'Task': t, 'Value': val, 
                     'L1': l1_v, 'Dist': dist_v, 'Beta': data['beta'], 'Travel': data['travel'], 'Temp': data['temp'], 'Alpha': data['alpha'], 'MinBlock': data['min_block']
                 })
                 
        # Generalization
        for t, val in data['gen_accs'].items():
            gen_records.append({
                'Run': run, 'Source': source, 'Type': r_type, 'Weight': w, 'Task': t, 'Value': val, 
                'L1': l1_v, 'Dist': dist_v, 'Beta': data['beta'], 'Travel': data['travel'], 'Temp': data['temp'], 'Alpha': data['alpha'], 'MinBlock': data['min_block']
            })
            
        # Weight Evolution
        for rec in data['weight_history']:
            weight_records.append({
                'Run': run, 
                'Source': source,
                'Type': r_type, 
                'Weight': w, 
                'Step': rec['step'], 
                'L1_Sum': rec['l1_sum']
            })

        # Aggregated Metrics per Run
        all_accs = list(data['final_accs'].values())
        if all_accs:
            mean_acc = np.mean(all_accs)
            
            # Fraction solved (>= 0.8)
            n_solved = sum(1 for a in all_accs if a >= 0.8)
            frac_solved = n_solved / len(all_accs)
            
            agg_records.append({
                'Run': run,
                'Source': source,
                'Type': r_type,
                'Weight': w,
                'L1': l1_v,
                'Dist': dist_v,
                'Beta': data['beta'],
                'Travel': data['travel'],
                'Temp': data['temp'],
                'Alpha': data['alpha'],
                'MinBlock': data['min_block'],
                'Switch_Rate': data['switch_rate'],
                'Entropy': data['entropy'],
                'Mean_Accuracy': mean_acc,
                'Fraction_Solved': frac_solved,
                'Mean_Speed': data['mean_speed']
            })


    # Prepare output dir (needed by all modes)
    first_sweep = sweep_dirs[0]
    if len(sweep_dirs) == 1:
        out_dir = first_sweep / "comparison_plots"
    else:
        out_dir = first_sweep.parent / "comparison_plots"
    out_dir.mkdir(exist_ok=True)

    print(f"Outputting plots to: {out_dir}", flush=True)

    # In targeted mode skip all population-level plotting and jump straight to submission
    if args.mode == "targeted":
        _run_targeted(args, run_dirs, out_dir)
        return
    
    # DEBUG: Check Found Parameter Values
    # This helps explain why some plots might be missing rows/cols (e.g. Travel=500 missing)
    unique_travel = sorted(list(set(r['Travel'] for r in agg_records)))
    unique_beta = sorted(list(set(r['Beta'] for r in agg_records)))
    unique_minblock = sorted(list(set(r.get('MinBlock', 0) for r in agg_records)))
    unique_alpha = sorted(list(set(r.get('Alpha', 0) for r in agg_records)))
    
    print("\n" + "="*40)
    print("DEBUG: Unique Parameter Values Found in Logs")
    print(f"  Travel:   {unique_travel}")
    print(f"  Beta:     {unique_beta}")
    print(f"  MinBlock: {unique_minblock}")
    print(f"  Alpha:    {unique_alpha}")
    print("="*40 + "\n")
    
    # Define Palette
    palette = {
        "Baseline": "gray",
        "L1": "tab:blue",
        "Proximal L1": "tab:red",
        "Distance": "tab:green",
    }

    def filter_best_models(records, group_cols, value_col="Value", maximize=True):
        """
        Retain only the BEST run for each group in group_cols.
        If multiple runs exist in a group (varying by unlisted params), pick the one with best mean value.
        """
        if not records: return []
        df = pd.DataFrame(records)
        
        # Check columns
        for c in group_cols:
            if c not in df.columns:
                print(f"Warning: Group col {c} not in records, skipping filtering.")
                return records
                
        # We need to Aggregate per RUN first (if records are per-task) is handled?
        # If records are per-task (e.g. perf_records), we need to find best RUN based on MEAN Value.
        
        # 1. Calc Mean Score per Run
        run_scores = df.groupby('Run')[value_col].mean().reset_index(name='RunScore')
        
        # 2. Merge Score back
        df = df.merge(run_scores, on='Run')
        
        # 3. Identify Best Run per Group
        # We want the RunID that maximizes RunScore for each (Source, Type, Weight...) group
        # Sort so we can take first
        df_sorted = df.sort_values(by=group_cols + ['RunScore'], ascending=[True]*len(group_cols) + [not maximize])
        
        # Drop duplicates on the Group Columns -> keeps the first (best)
        # But we need to keep ALL records for that Best Run.
        best_runs_df = df_sorted.drop_duplicates(subset=group_cols, keep='first')
        best_run_ids = best_runs_df['Run'].unique()
        
        # 4. Filter original records
        # Only keep records where Run is in best_run_ids
        filtered = [r for r in records if r['Run'] in best_run_ids]
        print(f"Filtered {len(records)} -> {len(filtered)} records (Best Models Only).")
        return filtered


    
    def plot_metric_simple(records, filename, title, ylabel, value_col="Value"):
        """Vertical subplots for simple comparison (one plot per run source)."""
        if not records:
            return

        # --- KEY CHANGE: Filter for BEST models (maximizing accuracy) ---
        # Group by: Source, Type, Weight. 
        # Varying: Beta, Alpha, Temp, Travel, MinBlock.
        # We want the best combination of those varying params.
        records = filter_best_models(records, group_cols=['Source', 'Type', 'Weight'], value_col=value_col)
            
        df = pd.DataFrame(records)
        df = df.sort_values(by='Weight')
        
        # Backward compatibility for 'Value' column name
        if value_col not in df.columns and 'Value' in df.columns:
            value_col = 'Value'
            
        # Identify sources and sort them
        sources = sorted(df['Source'].unique())
        n_plots = len(sources)
        
        if n_plots == 0: return

        # Create vertical subplots
        fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(10, 6 * n_plots), sharex=True)
        if n_plots == 1: axes = [axes] # Ensure iterable
        
        for ax, source in zip(axes, sources):
            df_curr = df[df['Source'] == source]
            
            # Determine if dodging is needed (Seaborn crashes if dodge=True but only 1 hue)
            n_hue = len(df_curr['Type'].unique())
            curr_dodge = 0.3 if n_hue > 1 else 0.0

            # Pointplot for means (averaged over tasks within a single run, i.e. general trend)
            sns.pointplot(data=df_curr, x="Weight", y=value_col, hue="Type", 
                          dodge=curr_dodge, markers="o", linestyles="-", errwidth=1.5, capsize=0.1, alpha=0.8,
                          palette=palette, ax=ax)
            
            # Stripplot for individual dots (tasks)
            sns.stripplot(data=df_curr, x="Weight", y=value_col, hue="Type", 
                          dodge=curr_dodge, alpha=0.4, jitter=True, legend=False,
                          palette=palette, ax=ax)
            
            ax.set_title(f"{title} - {source}")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Regularization Weight (Primary)")
            
            # Format X-axis labels on the last plot
            new_labels = []
            for l in ax.get_xticklabels():
                try:
                    val = float(l.get_text())
                    new_labels.append(f"{val:.1e}")
                except ValueError:
                    new_labels.append(l.get_text())
            ax.set_xticklabels(new_labels, rotation=45, ha='right')
            
            # Only show legend on first plot or outside? Let's keep it on each for clarity or handle it.
            # Pointplot puts legend on axis by default.
        
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()
        print(f"Saved {filename}")

    # Plot Weights Evolution - Grouped by Type
    if weight_records:
        df_w = pd.DataFrame(weight_records)
        plt.figure(figsize=(12, 7))
        
        # Plot lines: Group by Run (units) but Color by Type
        sns.lineplot(data=df_w, x="Step", y="L1_Sum", 
                     hue="Type", units="Run", estimator=None, 
                     alpha=0.6, linewidth=1.5, palette=palette)
        
        # Annotate end of lines
        # Group by Run to find the last point for each line
        # Assuming each Run corresponds to one line
        for run_id, group in df_w.groupby("Run"):
            # Find last step
            last_pt = group.loc[group['Step'].idxmax()]
            x_pos = last_pt['Step']
            y_pos = last_pt['L1_Sum']
            weight_val = last_pt['Weight']
            
            # Format annotation
            label_text = f"{weight_val:.1e}"
            
            # Add text
            plt.text(x_pos + 100, y_pos, label_text, 
                     horizontalalignment='left', 
                     verticalalignment='center',
                     fontsize=9, color='black', alpha=0.8)
        
        plt.title("Weight Evolution: Sparsity/Norm over Time")
        plt.ylabel("Sum |W_rec|")
        plt.xlabel("Training Step")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Reg Type")
        plt.tight_layout()
        plt.savefig(out_dir / "weights_evolution.png")
        plt.close()
        print(f"Saved weights_evolution.png")

    # --- 1. Simple Comparison: L1 vs Distance (No Combination) ---
    print("Plotting Simple Comparisons (L1 vs Dist)...")
    simple_types = ["Baseline", "L1", "Distance", "Proximal L1"]
    
    def filter_simple(recs):
        return [r for r in recs if r.get('Type') in simple_types]

    try:
        plot_metric_simple(filter_simple(perf_records), "simple_perf.png", "Performance: L1 vs Distance", "Accuracy")
        plot_metric_simple(filter_simple(speed_records), "simple_speed.png", "Speed: L1 vs Distance", "Steps")
        plot_metric_simple(filter_simple(gen_records), "simple_gen.png", "Generalization: L1 vs Distance", "Accuracy")
        plot_metric_simple(filter_simple(agg_records), "simple_frac_solved.png", "Fraction Solved: L1 vs Distance", "Fraction", value_col="Fraction_Solved")
    except Exception as e:
        print(f"ERROR in Simple Plotting: {e}")

    # --- 2. Heatmaps for L1 + Distance ---
    print("Plotting Heatmaps (L1 + Dist)...")
    def plot_heatmap_l1_dist(records, filename, title, value_col):
        """Heatmap for L1 + Distance models."""
        try:
            if not records:
                return
                
            df = pd.DataFrame(records)
            # Filter for combined models
            if 'Type' not in df.columns:
                 print(f"Skipping Heatmap {filename}: Missing 'Type'")
                 return
                 
            df = df[df['Type'] == "L1 + Distance"]
            if df.empty:
                return
    
            # Pivot: L1 (y), Dist (x), Value (z)
            # Max across all runs/sources (Best Model)
            pivot = df.pivot_table(index="L1", columns="Dist", values=value_col, aggfunc='max')
            
            # Sort index/cols to ensure correct order
            pivot = pivot.sort_index(ascending=False) # High L1 at top
            pivot = pivot.sort_index(axis=1, ascending=True)
            
            # Format Ticks
            y_labels = [f"{v:.1e}" for v in pivot.index]
            x_labels = [f"{v:.1e}" for v in pivot.columns]
    
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues_r",  
                        xticklabels=x_labels, yticklabels=y_labels)
            
            plt.title(title)
            plt.xlabel("Distance Weight")
            plt.ylabel("L1 Weight")
            plt.tight_layout()
            plt.savefig(out_dir / filename)
            plt.close()
            print(f"Saved {filename}")
        except Exception as e:
            print(f"ERROR in Heatmap {filename}: {e}")

    # Call Heatmaps
    plot_heatmap_l1_dist(agg_records, "heatmap_mean_acc.png", "Mean Accuracy (L1 + Distance)", "Mean_Accuracy")
    plot_heatmap_l1_dist(agg_records, "heatmap_frac_solved.png", "Fraction Solved (L1 + Distance)", "Fraction_Solved")

    # --- 3. Foraging Landscape Plots (New!) ---
    # FacetGrid of Heatmaps:
    # Rows: Reg Type
    # Cols: Temperature
    # X: Travel Steps
    # Y: Beta Global
    # Color: Mean Accuracy
    
    print("Plotting Foraging Landscape FacetGrids...")
    
    def plot_foraging_landscape(records, filename, title, value_col, x_col, y_col):
        """Generates a FacetGrid of Heatmaps."""
        if not records: return
        df = pd.DataFrame(records)
        
        # Check if columns exist
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Skipping {filename}: Missing columns {x_col} or {y_col}")
            return

        # Using FacetGrid + Custom mapper
        def draw_heatmap(data, x, y, values, **kwargs):
            # Pivot local data
            if data.empty: return
            # aggfunc='max' ensures we pick the BEST model for this cell
            # i.e. if MinBlock varies inside this cell, we take the max accuracy
            piv = data.pivot_table(index=y, columns=x, values=values, aggfunc='max')
            # Sort for axes
            piv = piv.sort_index(ascending=False) 
            
            sns.heatmap(piv, annot=True, fmt=".2f", cmap="viridis", cbar=False, **kwargs)
            
        g = sns.FacetGrid(df, row="Type", col="Temp", margin_titles=True, height=4, aspect=1.2)
        g.map_dataframe(draw_heatmap, x=x_col, y=y_col, values=value_col)
        
        g.set_axis_labels(x_col, y_col)
        g.fig.suptitle(title, y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()
        print(f"Saved {filename}")

    # Plot Accuracy Landscape
    # Plot Accuracy Landscape
    try:
        # 1. Travel vs Beta (Existing)
        plot_foraging_landscape(agg_records, "forage_landscape_acc_trav_beta.png", "Accuracy: Travel vs Beta", "Mean_Accuracy", x_col="Travel", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_entr_trav_beta.png", "Entropy: Travel vs Beta", "Entropy", x_col="Travel", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_swt_trav_beta.png", "Switch: Travel vs Beta", "Switch_Rate", x_col="Travel", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_spd_trav_beta.png", "Speed: Travel vs Beta", "Mean_Speed", x_col="Travel", y_col="Beta")

        # 2. MinBlock vs Beta
        plot_foraging_landscape(agg_records, "forage_landscape_acc_blk_beta.png", "Accuracy: MinBlock vs Beta", "Mean_Accuracy", x_col="MinBlock", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_entr_blk_beta.png", "Entropy: MinBlock vs Beta", "Entropy", x_col="MinBlock", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_swt_blk_beta.png", "Switch: MinBlock vs Beta", "Switch_Rate", x_col="MinBlock", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_spd_blk_beta.png", "Speed: MinBlock vs Beta", "Mean_Speed", x_col="MinBlock", y_col="Beta")
        
        # 3. Alpha vs Beta
        plot_foraging_landscape(agg_records, "forage_landscape_acc_alpha_beta.png", "Accuracy: Alpha vs Beta", "Mean_Accuracy", x_col="Alpha", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_entr_alpha_beta.png", "Entropy: Alpha vs Beta", "Entropy", x_col="Alpha", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_swt_alpha_beta.png", "Switch: Alpha vs Beta", "Switch_Rate", x_col="Alpha", y_col="Beta")
        plot_foraging_landscape(agg_records, "forage_landscape_spd_alpha_beta.png", "Speed: Alpha vs Beta", "Mean_Speed", x_col="Alpha", y_col="Beta")

        # 4. Travel vs MinBlock
        plot_foraging_landscape(agg_records, "forage_landscape_acc_trav_blk.png", "Accuracy: Travel vs MinBlock", "Mean_Accuracy", x_col="Travel", y_col="MinBlock")
        plot_foraging_landscape(agg_records, "forage_landscape_entr_trav_blk.png", "Entropy: Travel vs MinBlock", "Entropy", x_col="Travel", y_col="MinBlock")
        plot_foraging_landscape(agg_records, "forage_landscape_swt_trav_blk.png", "Switch: Travel vs MinBlock", "Switch_Rate", x_col="Travel", y_col="MinBlock")
        plot_foraging_landscape(agg_records, "forage_landscape_spd_trav_blk.png", "Speed: Travel vs MinBlock", "Mean_Speed", x_col="Travel", y_col="MinBlock")

    except Exception as e:
        print(f"Error plotting Foraging Landscape: {e}")

    def plot_foraging_parameter_trends(records, out_dir):
        """Shows how performance varies with each foraging parameter individually."""
        if not records: return
        
        # --- KEY CHANGE: Filter for BEST models ---
        # But here the grouping changes per parameter loop!
        # If param is 'Beta', we Group By ['Source', 'Type', 'Beta'] and max over others.
        
        df_raw = pd.DataFrame(records)
        
        for param in ["Beta", "Travel", "Temp", "MinBlock"]:
            if param not in df_raw.columns: continue
            
            # Filter specific to this view
            filtered = filter_best_models(records, group_cols=['Source', 'Type', param], value_col="Mean_Accuracy")
            df = pd.DataFrame(filtered)
            
            plt.figure(figsize=(10, 6))
            # Use estimator=None to show the actual best run per X value (no averaging)
            # If multiple sources, we see multiple lines (good!)
            sns.lineplot(data=df, x=param, y="Mean_Accuracy", hue="Type", style="Source", 
                         markers=True, dashes=False, estimator=None, palette=palette, alpha=0.8)
            
            plt.title(f"Effect of {param} on Performance (Best Model per Value)")
            plt.ylabel("Mean Accuracy (Best Model)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"forage_trend_{param.lower()}.png")
            plt.close()
            print(f"Saved forage_trend_{param.lower()}.png")

    try:
        plot_foraging_parameter_trends(agg_records, out_dir)
    except Exception as e:
        print(f"Error plotting Foraging Trends: {e}")

    # Plot Interaction: Accuracy vs Switch Rate (Pareto)
    try:
        if agg_records:
            df = pd.DataFrame(agg_records)
            
            # Identify Best Runs
            # Group by Type and find the single best run overall
            # (Or perhaps Pareto frontier? But "Best Model" usually implies max accuracy)
            best_runs_df = pd.DataFrame(filter_best_models(agg_records, group_cols=['Source', 'Type'], value_col="Mean_Accuracy"))
            
            plt.figure(figsize=(12, 8))
            
            # 1. Plot ALL runs as faint background
            # Define markers for Temperature explicitly to ensure all 3 show up
            # temp_vals = df['Temp'].unique()
            # markers_map = {t: m for t, m in zip(sorted(temp_vals), ['o', 'X', 's', 'D', '^'])}
            # Actually better to let seaborn handle it but force 'style' to be categorical? 
            # Or just pass markers=True to use default list?
            
            sns.scatterplot(data=df, x="Switch_Rate", y="Mean_Accuracy", hue="Type", style="Temp", 
                            s=50, alpha=0.3, palette=palette, legend=False, markers=True)
            
            # 2. Plot BEST runs prominently
            sns.scatterplot(data=best_runs_df, x="Switch_Rate", y="Mean_Accuracy", hue="Type", style="Temp", 
                            s=200, alpha=1.0, palette=palette, edgecolor='black', linewidth=1.5, markers=True)
            
            # Annotate Best Runs
            for idx, row in best_runs_df.iterrows():
                # Label with relevant params
                label = f"B={row['Beta']},Tr={row['Travel']}"
                if 'MinBlock' in row: label += f",Mb={row['MinBlock']}"
                
                plt.text(row['Switch_Rate']+0.2, row['Mean_Accuracy'], label, 
                         fontsize=8, alpha=0.9, fontweight='bold')
            
            plt.title("Pareto Front: Accuracy vs Exploration (Best Models Highlighted)")
            plt.xlabel("Switch Rate (Switches per 1k steps)")
            plt.ylabel("Mean Accuracy")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "forage_pareto.png")
            plt.close()
            print("Saved forage_pareto.png")
    except Exception as e:
         print(f"Error plotting Pareto: {e}")

    # --- 3. Best Model Comparison (Variability Plots) ---
    print("Plotting Best Models Variability...", flush=True)
    best_runs = []
    
    if not agg_records:
        print("WARNING: agg_records is empty! Skipping Best Model Analysis.")
    else:
        try:
            # Identify Best Run per Type (based on Fraction Solved, tie-breaker Mean Acc)
            df_agg = pd.DataFrame(agg_records)
            if 'Type' not in df_agg.columns:
                print(f"CRITICAL: df_agg missing Type! columns: {df_agg.columns}")
                
            # Always include ALL Baselines? Or just one? User said "Baseline". Usually denotes the group.
            # Let's include ALL Baseline runs to show variability if there are multiple.
            if 'Type' in df_agg.columns:
                # Iterate over ALL types including Baseline to find the best configuration for each
                for t in ["Baseline", "L1", "Distance", "L1 + Distance"]:
                    subset = df_agg[df_agg['Type'] == t]
                    if not subset.empty:
                        # Identify SINGLE Best Run across all params for this Type
                        # Sort by Mean Accuracy primarily
                        best_run_row = subset.sort_values(by=['Mean_Accuracy', 'Fraction_Solved'], ascending=False).iloc[0]
                        
                        # Extract defining hyperparameters
                        # We want all runs that match this specific configuration
                        # (Weight, L1, Dist, Beta, Travel, Temp, Alpha)
                        # Use isclose for floats
                        
                        b_w = best_run_row['Weight']
                        b_l1 = best_run_row['L1']
                        b_dist = best_run_row['Dist']
                        b_beta = best_run_row['Beta']
                        b_travel = best_run_row['Travel']
                        b_temp = best_run_row['Temp']
                        b_alpha = best_run_row['Alpha']
                        
                        # Find matches
                        matches = subset[
                            (np.isclose(subset['Weight'], b_w)) & 
                            (np.isclose(subset['L1'], b_l1)) & 
                            (np.isclose(subset['Dist'], b_dist)) &
                            (np.isclose(subset['Beta'], b_beta)) &
                            (np.isclose(subset['Travel'], b_travel)) &
                            (np.isclose(subset['Temp'], b_temp)) & 
                            (np.isclose(subset['Alpha'], b_alpha))
                        ]['Run'].unique().tolist()

                        best_runs.extend(matches)
                        print(f"[{t}] Best Global Config: W={b_w:.1e}, L1={b_l1:.1e}, Dist={b_dist:.1e}, Beta={b_beta}, Trav={b_travel}, Temp={b_temp} (Acc={best_run_row['Mean_Accuracy']:.3f}). Best Runs: {matches}", flush=True)
        except Exception as e:
            print(f"ERROR in Best Model Analysis: {e}")
            
            
    # Create map for fast lookup
    run_name_to_path = {r_path.name: r_path for r_path, _ in run_dirs}

    # --- TARGETED ANALYSIS ---
    if args.mode == "targeted":
        if not args.target_runs:
             print("Error: --mode targeted requires --target-runs list.")
             sys.exit(1)
        
        print(f"Filtering for TARGETED runs containing: {args.target_runs}")
        # Find runs matching the target strings
        # logic: if target is a number, match strict ID. Else, substring.
        target_matches = []
        import re
        
        for r_name in run_name_to_path.keys():
            # Extract Run ID
            m = re.search(r"^run_(\d+)_", r_name)
            run_id = int(m.group(1)) if m else None
            
            match_found = False
            for t in args.target_runs:
                # Check if target is an integer ID
                if t.isdigit():
                    if run_id is not None and int(t) == run_id:
                        match_found = True
                        break
                else:
                    # Fallback to substring
                    if t in r_name:
                        match_found = True
                        break
            
            if match_found:
                 target_matches.append(r_name)
        
        target_run_ids = list(set(target_matches))
        print(f"Found {len(target_run_ids)} targeted runs: {target_run_ids}")
    else:
        target_run_ids = []

    # --- LAZY EVALUATION FOR WINNERS ---
    # Ensure all "Best Runs" have generalization data
    # Only run locally if we are in local mode. 
    # In cluster mode, we skip this check because we are about to submit jobs for it.
    if args.execution_mode == 'local' and args.compute_gen and args.mode in ["winners", "targeted"]:
        print("Checking/Running Generalization for Best Models...", flush=True)
        # heldout is defined earlier in main
        
    # --- LAZY EVALUATION ---
    # Determine which set of runs to evaluate
    if args.mode == "targeted":
        runs_to_eval = target_run_ids
    elif args.mode == "winners":
        runs_to_eval = best_runs
    else:
        runs_to_eval = [] # 'all' and 'none' handled differently or implicit
        
    if args.execution_mode == 'local' and args.compute_gen and runs_to_eval:
        print("Checking/Running Generalization for Selected Models...", flush=True)
        # heldout is defined earlier in main
        
        for r_name in runs_to_eval:
            if r_name in run_name_to_path:
                r_path = run_name_to_path[r_name]
                print(f"Lazy Eval Check: {r_name}", flush=True)
                run_evaluation_if_needed(r_path, heldout, method=args.gen_method, steps=args.probe_steps)
            else:
                print(f"WARNING: Could not find path for best run {r_name}")

        # Reload data for best runs (in case it was just generated)
        # This is slightly inefficient (re-loading) but ensures we have the latest probe results
        if args.execution_mode == 'local':
            for i, rec in enumerate(gen_records):
                if rec['Run'] in best_runs:
                    # Reload this specific run's gen accs
                    r_name = rec['Run']
                    if r_name in run_name_to_path:
                        # quick reload
                        r_path = run_name_to_path[r_name]
                        # code duplication from load_run_data, but kept simple here
                        gen_accs = {}
                        if args.gen_method == "probe":
                            gpath = r_path / "probe_compositionality.json"
                        else:
                            gpath = r_path / "eval_generalization.json"
                            
                        if gpath.exists():
                            try:
                                with open(gpath, "r") as f:
                                    gdata = json.load(f)
                                if args.gen_method == "probe":
                                    for t, vals in gdata.items():
                                        if isinstance(vals, dict) and "final_acc" in vals:
                                            gen_accs[t] = vals["final_acc"]
                                else:
                                    gen_accs = {k: v for k, v in gdata.items() if v >= 0}
                            except:
                                pass
                        
                        # Update record (Value is per-task, so we might need to re-expand?)
                        pass 

    # RE-POPULATE gen_records for Best Runs Only
    # Because we might have just created the data!
    best_gen_rec = []
    
    # RE-POPULATE gen_records for Best Runs Only
    # We attempt to load this data regardless of execution mode.
    # If the file exists (was local, or cluster job finished), we load it.
    if True: # Always attempt to load
        # We interpret 'best_gen_rec' freshly from disk for the best runs
        for r_name in best_runs:
            if r_name in run_name_to_path:
                r_path = run_name_to_path[r_name]
                # Load Type
                # Find the type from existing records?
                # Scan perf_records to find type
                rtype = "Unknown"
                for p in perf_records:
                    if p['Run'] == r_name:
                        rtype = p['Type']
                        break
                
                # Load Gen Data
                g_accs = {}
                if args.gen_method == "probe":
                    gpath = r_path / "probe_compositionality.json"
                else:
                    gpath = r_path / "eval_generalization.json"
                
                if gpath.exists():
                    try:
                        with open(gpath, "r") as f:
                            gdata = json.load(f)
                        if args.gen_method == "probe":
                            for t, vals in gdata.items():
                                if isinstance(vals, dict) and "final_acc" in vals:
                                    g_accs[t] = vals["final_acc"]
                        else:
                            g_accs = {k: v for k, v in gdata.items() if v >= 0}
                    except:
                        pass
                
                for t, v in g_accs.items():
                    best_gen_rec.append({
                        "Run": r_name,
                        "Type": rtype,
                        "Task": t,
                        "Value": v
                    })

    # --- PROBE LEARNING CURVE PLOTS FOR BEST MODELS ---
    # Generate learning curve plots if using probe method on winners
    # We run this check regardless of execution mode, as long as data exists
    if args.mode == "winners" and args.gen_method == "probe":
        print("\n[+] Generating Probe Learning Curve Plots for Best Models...")
    # Generate learning curve plots if using probe method on winners (or targets)
    if args.mode in ["winners", "targeted"] and args.gen_method == "probe":
        print("\n[+] Generating Probe Learning Curve Plots for Selected Models...")
        # If targeted, we probably want to see curves for targets?
        # If winners, we want winners.
        # User request: "plots" (plural) usually implies the comparison plots, but LC is per-run.
        # Let's plot LC for whichever set we are focusing on for 'execution'.
        # Re-use runs_to_eval logic if possible, or re-derive.
        lc_targets = best_runs if args.mode == "winners" else target_run_ids
        
        for r_name in lc_targets:
            if r_name in run_name_to_path:
                r_path = run_name_to_path[r_name]
                analysis_out = out_dir / f"{r_name}_analysis"
                analysis_out.mkdir(exist_ok=True, parents=True)
                
                plot_probe_learning_curves(r_path, analysis_out, args.gen_method)

    # Generalized Plotter for Best Models (Type on X axis)
    best_perf_rec = [r for r in perf_records if r['Run'] in best_runs]
    best_speed_rec = [r for r in speed_records if r['Run'] in best_runs]
    best_agg_rec = [r for r in agg_records if r['Run'] in best_runs]

    def plot_best_strip(records, filename, ylabel, title, value_col="Value"):
        if not records: return
        df = pd.DataFrame(records)
        
        # Ensure value exists
        if value_col not in df.columns and "Value" in df.columns:
            value_col = "Value"
            
        # Per-Task Averaging Logic
        has_task = ('Task' in df.columns)
        
        if has_task:
            grouped = df.groupby(['Type', 'Task'])[value_col].agg(['mean', 'sem']).reset_index()
        else:
            grouped = df
            pass

        plt.figure(figsize=(8, 6))
        n_hue = len(df['Type'].unique())
        curr_dodge = 0.3 if n_hue > 1 else 0.0

        if has_task:
            sns.stripplot(data=grouped, x="Type", y="mean", hue="Type", jitter=True, alpha=0.5, palette=palette, legend=False, dodge=curr_dodge)
            sns.pointplot(data=grouped, x="Type", y="mean", estimator=np.mean, 
                          color="black", markers="_", errorbar='se',
                          capsize=0.15, err_kws={'linewidth': 1.2}, linestyles='') 
            sns.pointplot(data=grouped, x="Type", y="mean", estimator=np.mean, 
                          color="black", markers="D", errorbar='se',
                          capsize=0.1, err_kws={'linewidth': 1.0}, linestyle='none')
        else:
            sns.stripplot(data=df, x="Type", y=value_col, hue="Type", jitter=True, alpha=0.5, palette=palette, legend=False, dodge=curr_dodge)
            sns.pointplot(data=df, x="Type", y=value_col, estimator=np.mean, 
                          color="black", markers="_", errorbar='se',
                          capsize=0.15, err_kws={'linewidth': 1.2}, linestyles='') 
            sns.pointplot(data=df, x="Type", y=value_col, estimator=np.mean, 
                          color="black", markers="D", errorbar='se',
                          capsize=0.1, err_kws={'linewidth': 1.0}, linestyle='none')
        
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()
        print(f"Saved {filename}")

    # Plot Best Per-Task Metrics
    plot_best_strip(best_perf_rec, "best_models_perf.png", "Accuracy", "Final Training Performance (Best Models)")
    plot_best_strip(best_speed_rec, "best_models_speed.png", "Steps", "Training Speed to 80% (Best Models)")
    plot_best_strip(best_gen_rec, "best_models_gen.png", "Accuracy", "Generalization Accuracy (Best Models)")

    # Plot Best Aggregated Metrics
    plot_best_strip(best_agg_rec, "best_models_frac_solved.png", "Fraction Solved", "Fraction of Tasks Solved (Best Models)", value_col="Fraction_Solved")
    
    # Also reuse the stacked bar plot for these best models
    plot_task_success_counts(best_perf_rec, out_dir, palette, filename="best_models_task_success_counts_stacked.png")

    # --- NEW: Best Models Gen Accuracy vs Task (Grouped Bar Plot) ---
    def plot_gen_acc_task_comparison(records, filename):
        if not records: return
        df = pd.DataFrame(records)
        if 'Task' not in df.columns: return

        plt.figure(figsize=(12, 7))
        # Bar Plot: X=Task, Y=Value, Hue=Type
        
        sns.barplot(data=df, x="Task", y="Value", hue="Type", 
                    palette=palette, errorbar='se', capsize=0.1, alpha=0.8)
        
        plt.title("Generalization Accuracy per Task (Best Models Comparison)")
        plt.ylabel("Accuracy")
        plt.xlabel("Task")
        plt.ylim(0, 1.05)
        # Removed chance line as requested
        # plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Reg Type")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()
        print(f"Saved {filename}")

    if best_gen_rec:
        plot_gen_acc_task_comparison(best_gen_rec, "best_models_gen_acc_per_task.png")

    # Identify target runs based on gen_mode
    target_paths = []
    
    # Helper to find path for a given run name
    def get_path_for_run(r_name):
        return run_name_to_path.get(r_name)

    if args.mode == "winners":
        target_paths = [run_name_to_path[r] for r in best_runs if r in run_name_to_path]
    elif args.mode == "targeted":
        target_paths = [run_name_to_path[r] for r in target_run_ids if r in run_name_to_path]
    elif args.mode == "all":
        # Collect ALL runs observed
        # perf_records contains all 'Run' values processed
        all_seen = set(r['Run'] for r in agg_records)
        target_paths = [run_name_to_path[r] for r in all_seen if r in run_name_to_path]
    elif args.mode == "none":
        target_paths = []

    # Run Analysis (Local or Cluster) on the selected target set
    if target_paths:
        run_analysis_on_models(target_paths, out_dir, args)

    # Plot Weight Evolution: Highlight Best Models
    if weight_records:
        df_w = pd.DataFrame(weight_records)
        best_w_recs = [r for r in weight_records if r['Run'] in best_runs]
        
        # 1. Highlight Plot (Existing)
        plt.figure(figsize=(10, 6))
        # Use style="Source" to visually distinguish runs from different sweeps
        sns.lineplot(data=df_w, x="Step", y="L1_Sum", 
                     hue="Type", style="Source", units="Run", estimator=None, 
                     alpha=0.15, linewidth=1, palette=palette, legend=None)
        
        if best_w_recs:
            df_best = pd.DataFrame(best_w_recs)
            # Highlight best trajectories
            # Use same style mapping for consistency
            sns.lineplot(data=df_best, x="Step", y="L1_Sum", 
                         hue="Type", style="Source", units="Run", estimator=None, 
                         alpha=1.0, linewidth=2.5, palette=palette)
            
            # Annotate
            for run_id, group in df_best.groupby("Run"):
                last_pt = group.loc[group['Step'].idxmax()]
                label_text = f"{last_pt['Weight']:.1e}"
                plt.text(last_pt['Step'] + 100, last_pt['L1_Sum'], label_text, 
                         ha='left', va='center', fontsize=9, fontweight='bold', alpha=1.0)
 
        plt.title("Weight Evolution: Sparsity/Norm (Highlighted Best Averaged Configs)")
        plt.ylabel("Sum |W_rec|")
        plt.xlabel("Training Step")
        
        # Legend: Seaborn creates a combined legend for Hue and Style automatically
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(out_dir / "weights_evolution.png")
        plt.close()
        print("Saved weights_evolution.png")

        # 2. NEW: Best Models Only (No Baseline)
        if best_w_recs:
             # Filter out Baseline
             df_best_nobase = df_best[df_best['Type'] != "Baseline"]
             
             if not df_best_nobase.empty:
                 plt.figure(figsize=(10, 6))
                 sns.lineplot(data=df_best_nobase, x="Step", y="L1_Sum", 
                              hue="Type", style="Source", units="Run", estimator=None, 
                              alpha=1.0, linewidth=2.5, palette=palette)
                 
                 # Annotate
                 for run_id, group in df_best_nobase.groupby("Run"):
                    last_pt = group.loc[group['Step'].idxmax()]
                    label_text = f"{last_pt['Weight']:.1e}"
                    plt.text(last_pt['Step'] + 100, last_pt['L1_Sum'], label_text, 
                             ha='left', va='center', fontsize=9, fontweight='bold', alpha=1.0)
                 
                 plt.title("Weight Evolution: Best Models (No Baseline)")
                 plt.ylabel("Sum |W_rec|")
                 plt.xlabel("Training Step")
                 plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                 plt.tight_layout()
                 plt.savefig(out_dir / "weights_evolution_best_nobaseline.png")
                 plt.close()
                 print("Saved weights_evolution_best_nobaseline.png")

def plot_task_success_counts(records, out_dir, palette, filename="task_success_counts.png"):
    """
    Plot stacked bar chart of how many models reached >= 80% accuracy per task.
    X-axis: Tasks
    Y-axis: Count of models
    Colors: Model Type
    """
    if not records:
        return
        
    df = pd.DataFrame(records)
    
    # Identify unique tasks to ensure we show all, even if 0 count
    all_tasks = sorted(df['Task'].unique())
    
    # Filter for success
    success_df = df[df['Value'] >= 0.8]
    
    if success_df.empty:
        print("No models reached 80% accuracy on any task.")
        return

    # Count successes: Run is unique ID, but we want count of Runs per (Task, Type)
    # Group by Task, Type -> count unique Runs
    counts = success_df.groupby(['Task', 'Type'])['Run'].nunique().reset_index(name='Count')
    
    # Pivot: Index=Task, Columns=Type, Values=Count
    pivot_df = counts.pivot(index='Task', columns='Type', values='Count').fillna(0)
    
    # Reindex tasks to ensure all are present and sorted
    pivot_df = pivot_df.reindex(all_tasks, fill_value=0)
    
    # Reorder columns to match logical order if possible
    desired_order = ["Baseline", "L1", "Proximal L1", "Distance", "L1 + Distance"]
    existing_cols = [c for c in desired_order if c in pivot_df.columns]
    # Add any others that might exist but aren't in desired_order
    remaining = [c for c in pivot_df.columns if c not in desired_order]
    final_cols = existing_cols + remaining
    
    pivot_df = pivot_df[final_cols]
    
    # Colors
    colors = [palette.get(c, "black") for c in pivot_df.columns]
    
    # Plot
    # Note: Pandas plot returns axes, but we want to control figure.
    # We can create figure first, then pass ax.
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_df.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.8)
    
    ax.set_title("Number of Models Reaching 80% Performance per Task")
    ax.set_ylabel("Count of Models")
    ax.set_xlabel("Task")
    # Rotate ticks
    plt.xticks(rotation=45, ha='right')
    ax.legend(title="Reg Type")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    out_path = out_dir / filename
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path.name}")



if __name__ == "__main__":
    main()

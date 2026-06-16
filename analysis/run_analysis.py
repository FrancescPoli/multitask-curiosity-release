#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main analysis runner for Multitask Curiosity.

Usage:
    python -m analysis.run_analysis --run_dir logs/my_run [options]

Options:
    --plot_training       : Plot global training curves (loss/acc) [Default: True]
    --plot_foraging       : Plot foraging dynamics (rho/P, decisions, perf) [Default: True]
    --plot_weights        : Plot Wrec heatmap and histogram [Default: True]
    --plot_checkpoints    : Plot Wrec density/L1 over checkpoints [Default: True]
    --plot_scaffolding    : Plot scaffolding task selection/schedule [Default: True]
    
    --no_plot_training    : Disable training plots
    # ... etc for other flags (use --no_ prefix to disable defaults)

    --step_range START END : Range of steps to plot for foraging/scaffolding (optional)
"""

import argparse
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt

# Ensure analysis/.. imports work if running as script from within analysis/
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "analysis"

from analysis.utils.paths import get_logs_dir

import analysis.plotting.utils as autils
import analysis.plotting.plots as aplots


def parse_args():
    parser = argparse.ArgumentParser(description="Run analysis and plots for a trained model.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory (logs/...)")
    
    # Plotting flags (Defaults to True)
    parser.add_argument("--plot_training", action=argparse.BooleanOptionalAction, default=True, help="Plot global training curves")
    parser.add_argument("--plot_foraging", action=argparse.BooleanOptionalAction, default=True, help="Plot foraging dynamics")
    parser.add_argument("--plot_weights", action=argparse.BooleanOptionalAction, default=True, help="Plot final weights")
    parser.add_argument("--plot_checkpoints", action=argparse.BooleanOptionalAction, default=True, help="Plot weight evolution over checkpoints")
    parser.add_argument("--plot_scaffolding", action=argparse.BooleanOptionalAction, default=True, help="Plot scaffolding schedule")

    # Options
    # Options
    parser.add_argument("--step_range", type=int, nargs=2, default=None, help="Start and End step for time-series plots")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots to (instead of showing them)")
    
    return parser.parse_args()



def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        # Try finding it in the central logs directory
        potential_path = get_logs_dir() / args.run_dir
        if potential_path.exists():
            run_dir = potential_path
            print(f"   Resolved run directory to: {run_dir}")
        else:
             # Try sweep subfolder commonly used
             potential_sweep = get_logs_dir() / "sweep" / args.run_dir
             if potential_sweep.exists():
                 run_dir = potential_sweep
                 print(f"   Resolved run directory to: {run_dir}")
    
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Saving plots to: {save_dir}")

    def show_or_save(fig, filename):
        if fig is None:
            return
        # If fig is a list (e.g. from plot_Wrec_and_hist), handle each
        if isinstance(fig, list):
            for i, f in enumerate(fig):
                show_or_save(f, f"{Path(filename).stem}_{i}{Path(filename).suffix}")
            return

        if args.save_dir:
            path = Path(args.save_dir) / filename
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved {filename}")
        else:
            plt.show()

    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"-> Analyzing run: {run_dir}")
    device = torch.device("cpu")

    # Load Meta & Model
    print("   Loading model metadata...")
    try:
        meta = autils.load_meta(run_dir)
    except Exception as e:
        print(f"   Warning: Could not load model_meta.json: {e}")
        meta = {}

    # Load Curriculum Data (CSV)
    print("   Loading curriculum data...")
    try:
        rows = autils.load_curriculum_rows(run_dir)
        # Also dataframe for scaffolding/switching plots
        df = autils.load_curriculum_dataframe(run_dir)
        has_data = True
    except FileNotFoundError:
        print("   Warning: run_curriculum.csv not found.")
        rows = []
        df = None
        has_data = False

    # Check regime
    is_foraging = False
    
    # 1. Try meta config
    if meta:
        cfg = meta.get("cfg", {})
        # Explicit regime key
        if cfg.get("regime") == "foraging":
            is_foraging = True
        # Or presence of "foraging" settings (e.g. "foraging": "mvt")
        elif cfg.get("foraging", "none") != "none":
             is_foraging = True

    # 2. Fallback to data if not yet confirmed
    if not is_foraging and has_data and df is not None:
        if "regime" in df.columns:
            is_foraging = (df["regime"] == "foraging").any()


    # =========================================================================
    # 1. Training Curves
    # =========================================================================
    if args.plot_training and has_data:
        print("\n[+] Plotting Training Curves...")
        loss_hist, acc_hist = autils.build_loss_acc_from_rows(rows)
        if loss_hist:
            fig = aplots.plot_loss_and_ema(loss_hist, acc_hist, title="Global Training")
            show_or_save(fig, "training_curves.png")
            
            # --- New: Per-task EMA plot (Requested by User) ---
            print("   Plotting Per-Task EMA Training Curves...")
            steps_pt, per_task_pt = autils.build_per_task_ema_dict(rows)
            
            # Smoothing for small batch sizes (Bio-Plausible)
            # If batch_size=1, raw acc is 0/1. We need strong smoothing.
            batch_size = 64
            if "cfg" in meta:
                try:
                    batch_size = int(meta["cfg"].get("batch_size", 64))
                except:
                    pass
            
            print(f"   [Smoother] Batch Size: {batch_size}")
            if batch_size <= 10 or is_foraging:
                print(f"   [Smoother] Applying strong smoothing (alpha=0.001) for online learning.")
                per_task_pt = autils.smooth_per_task_traces(per_task_pt, alpha=0.001)

            if steps_pt:
                fig_pt = aplots.plot_per_task_ema_gapped(
                    steps_pt, per_task_pt, 
                    title="Per-Task Training Accuracy (EMA)",
                    xlim=None, ylim=(0.0, 1.0)
                )
                show_or_save(fig_pt, "training_per_task.png")
            else:
                print("   No per-task EMA data available.")
        else:
            print("   No loss/acc data found.")

    # =========================================================================
    # 2. Weights (Final)
    # =========================================================================
    if args.plot_weights:
        print("\n[+] Plotting Final Weights...")
        try:
            # We need the model for this
            state = autils.load_state_dict_from_run(run_dir, map_location=device)
            model = autils.build_model_from_meta(meta, device=device)
            autils.load_state_into_model(model, state)
            
            W_rec = autils.get_Wrec(model)
            figs = aplots.plot_Wrec_and_hist(W_rec)
            show_or_save(figs, "Wrec_analysis.png") # will handle list
            
            brain_mask = autils.load_brain_mask(meta, model)
            if brain_mask is not None:
                fig = aplots.plot_brain_mask(brain_mask)
                show_or_save(fig, "brain_mask.png")
                
        except Exception as e:
            print(f"   Skipping weights plot: {e}")

    # =========================================================================
    # 3. Checkpoints (Weight Evolution)
    # =========================================================================
    if args.plot_checkpoints:
        print("\n[+] Plotting Weight Evolution (Checkpoints)...")
        try:
            # Check if checkpoints exist first to avoid lengthy model build if not needed
            ckpts = autils.list_checkpoint_steps_and_paths(run_dir)
            if ckpts:
                fig1 = aplots.plot_Wrec_density_over_checkpoints(run_dir, meta, device=device)
                show_or_save(fig1, "checkpoint_density.png")
                
                fig2 = aplots.plot_Wrec_L1_over_checkpoints(run_dir, meta, device=device)
                show_or_save(fig2, "checkpoint_L1.png")
            else:
                print("   No intermediate checkpoints found (state_step*.pt).")
        except Exception as e:
            print(f"   Skipping checkpoints plot: {e}")

    # =========================================================================
    # 4. Foraging Analysis
    # =========================================================================
    if args.plot_foraging:
        if is_foraging and has_data:
            print("\n[+] Plotting Foraging Dynamics...")
            steps_f, rho_hist, P_hist, action_hist, task_hist = autils.build_foraging_traces_from_rows(rows)
            
            if steps_f:
                # Get tasks
                tasks = []
                if "cfg" in meta and "tasks" in meta["cfg"]:
                    tasks = list(meta["cfg"]["tasks"])
                if not tasks:
                     tasks = sorted({t for t in task_hist if t and t != "TRAVEL"})

                fig1 = aplots.plot_foraging_rho_P(
                    steps_f, rho_hist, P_hist, task_hist, tasks,
                    step_range=args.step_range
                )
                show_or_save(fig1, "foraging_rho_P.png")
                
                fig2 = aplots.plot_foraging_decisions(
                    steps_f, action_hist,
                    step_range=args.step_range
                )
                show_or_save(fig2, "foraging_decisions.png")
                
                # Perf
                train_alpha = float(meta.get("cfg", {}).get("forage_alpha_local", 0.03))
                steps_p, perf_h, task_h_p = autils.build_foraging_perf_from_rows(rows, alpha=train_alpha)
                if steps_p:
                    fig3 = aplots.plot_foraging_per_task_perf(
                        steps_p, perf_h, task_h_p, tasks,
                        step_range=args.step_range
                    )
                    show_or_save(fig3, "foraging_perf.png")
            else:
                print("   No foraging traces found in data.")
        else:
            if args.plot_foraging and not is_foraging:
                # If specifically requested but not foraging, warn
                 print("\n🍎 Skipping Foraging Plots (Not a foraging run or no data).")

    # =========================================================================
    # 5. Scaffolding Analysis
    # =========================================================================
    if args.plot_scaffolding:
        if is_foraging and df is not None:
             print("\n[+] Plotting Scaffolding Analysis...")
             fig1 = aplots.plot_scaffolding_task_selection(df, step_range=args.step_range)
             show_or_save(fig1, "scaffolding_selection.png")
             
             fig2 = aplots.plot_scaffolding_first_appearance(df)
             show_or_save(fig2, "scaffolding_appearance.png")
             
             fig3, _ = aplots.plot_task_switching_analysis(df, step_range=args.step_range)
             show_or_save(fig3, "scaffolding_switching.png")
        else: 
             if args.plot_scaffolding and not is_foraging:
                  print("\n🏗️ Skipping Scaffolding Plots (Not a foraging run or no data).")

    print("\n[Done].")

if __name__ == "__main__":
    main()

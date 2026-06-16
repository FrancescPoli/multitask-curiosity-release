"""
Shared utility functions for analysis scripts.
Includes data loading, model extraction, and common helpers.
"""
from __future__ import annotations

import csv
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# We assume that the user's PYTHONPATH allows imports from curiosity
try:
    from curiosity.utils.model_loader import (
        load_meta,
        load_state_dict_from_run,
        build_model_from_meta,
        load_state_into_model
    )
except ImportError:
    # If running from analysis folder without properly set pythonpath, we might need this
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from curiosity.utils.model_loader import (
        load_meta,
        load_state_dict_from_run,
        build_model_from_meta,
        load_state_into_model
    )


# ---------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------

def _apply_axes_limits(
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Small helper to consistently set x/y limits if provided."""
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


# ---------------------------------------------------------------------
# Model / Weights Extraction
# ---------------------------------------------------------------------

def get_Wrec(model: nn.Module) -> torch.Tensor:
    """
    Extract recurrent weight matrix W_rec.

    Handles both:
      - separate-input model: model.cell_sep.Wrec
      - single-matrix model: model.cell.W[n_input:, :]
    """
    # Separate-input (Khona-style)
    if hasattr(model, "cell_sep") and getattr(model, "cell_sep") is not None:
        return model.cell_sep.Wrec

    # Single-matrix (Yang-style)
    if hasattr(model, "cell") and getattr(model, "cell") is not None:
        W_full = model.cell.W
        n_in = int(model.hp.n_input)
        return W_full[n_in:, :]

    raise RuntimeError("Could not locate recurrent weights (no cell_sep or cell found).")


def load_brain_mask(meta: Dict[str, Any], model: nn.Module) -> Optional[np.ndarray]:
    """
    Try to get a brain mask either from the model (buffer) or from mask_path in meta.

    Returns:
        np.ndarray or None if not available / load failed.
    """
    # 1) If the model already has a 'brain_mask' buffer, use that.
    if hasattr(model, "brain_mask"):
        bm = getattr(model, "brain_mask")
        if isinstance(bm, torch.Tensor):
            return bm.detach().cpu().numpy()

    # 2) Otherwise, see if mask_path was stored in meta
    mask_path = meta.get("model_kwargs", {}).get("mask_path", None)
    if not mask_path:
        return None

    mask_path = Path(mask_path)
    if not mask_path.is_absolute():
        # Ideally this should be relative to the run dir, but let's try relative to repo root or script
        # This part is tricky if we don't know where relative path starts from.
        # Original code assumed relative to the script location.
        # We will try to resolve it relative to the run_meta's presumed location if possible, 
        # but here we don't have run_dir passed in. 
        # We'll just try strict existence or relative to current cwd.
        if not mask_path.exists():
             # Fallback: check relative to this file's parent (analysis/)
             candidate = Path(__file__).resolve().parent / mask_path
             if candidate.exists():
                 mask_path = candidate

    if not mask_path.exists():
        # print(f"[warn] mask_path={mask_path} does not exist on disk; skipping mask plot.")
        return None

    try:
        arr = np.load(mask_path)
        return arr
    except Exception as e:
        print(f"[warn] Failed to load brain mask from {mask_path}: {e}")
        return None


def list_checkpoint_steps_and_paths(run_dir: Path) -> list[tuple[int, Path]]:
    """
    Return a sorted list of (step, path) for intermediate state_dict checkpoints.

    We expect files named like: state_step000100.pt, state_step000200.pt, ...
    """
    ckpts: list[tuple[int, Path]] = []
    if not run_dir.exists():
        return ckpts
        
    for p in run_dir.glob("state_step*.pt"):
        m = re.search(r"state_step(\d+)\.pt", p.name)
        if not m:
            continue
        step = int(m.group(1))
        ckpts.append((step, p))

    ckpts.sort(key=lambda sp: sp[0])
    return ckpts


# ---------------------------------------------------------------------
# Data Loading (CSV / DataFrame)
# ---------------------------------------------------------------------

def load_curriculum_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Load run_curriculum.csv from a run directory as a list of dicts.
    """
    csv_path = run_dir / "run_curriculum.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find run_curriculum.csv in {run_dir}")

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # sort by step to be safe
    def _step_of(r: Dict[str, Any]) -> int:
        try:
            return int(r.get("step", 0))
        except Exception:
            return 0

    rows.sort(key=_step_of)
    return rows


def load_curriculum_dataframe(run_dir: Path) -> pd.DataFrame:
    """
    Load run_curriculum.csv as a pandas DataFrame.
    Preprocesses 'step', 'rho', 'P' to numeric.
    """
    csv_path = run_dir / "run_curriculum.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find run_curriculum.csv in {run_dir}")
    
    df = pd.read_csv(csv_path)
    # Coerce numeric columns
    for col in ['step', 'rho', 'P', 'decision', 'loss', 'acc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df


def build_loss_acc_from_rows(rows: List[Dict[str, Any]]) -> tuple[list[float], list[float]]:
    """
    Reconstruct global loss_hist and acc_hist from curriculum rows.
    """
    loss_hist: List[float] = []
    acc_hist: List[float] = []

    for r in rows:
        mode = r.get("mode", "")
        if mode == "travel":
            continue

        loss_str = r.get("loss", "")
        if loss_str in ("", None):
            continue

        try:
            loss_val = float(loss_str)
        except ValueError:
            continue

        acc_str = r.get("acc", "")
        if acc_str not in ("", None):
            try:
                acc_val = float(acc_str)
            except ValueError:
                acc_val = acc_hist[-1] if acc_hist else 0.0
        else:
            acc_val = acc_hist[-1] if acc_hist else 0.0

        loss_hist.append(loss_val)
        acc_hist.append(acc_val)

    return loss_hist, acc_hist


def build_foraging_traces_from_rows(
    rows: List[Dict[str, Any]]
) -> tuple[list[int], list[float], list[float], list[int], list[str]]:
    """
    Extract foraging-specific traces (rho, P, decisions) from curriculum rows.
    """
    steps_f: List[int] = []
    rho_hist: List[float] = []
    P_hist: List[float] = []
    action_hist: List[int] = []
    task_hist: List[str] = []

    for r in rows:
        regime = r.get("regime", "")
        if regime != "foraging":
            continue

        step_str = r.get("step", "0")
        rho_str = r.get("rho", "")
        P_str = r.get("P", "")
        decision_str = r.get("decision", "0")

        if rho_str in ("", None) or P_str in ("", None):
            continue

        try:
            step = int(step_str)
            rho_val = float(rho_str)
            P_val = float(P_str)
            decision = int(decision_str)
        except ValueError:
            continue

        mode = r.get("mode", "")
        if mode == "travel":
            task = "TRAVEL"
        else:
            task = r.get("task", "")

        steps_f.append(step)
        rho_hist.append(rho_val)
        P_hist.append(P_val)
        action_hist.append(decision)
        task_hist.append(task)

    return steps_f, rho_hist, P_hist, action_hist, task_hist


def build_foraging_perf_from_rows(
    rows: List[Dict[str, Any]],
    alpha: float = 0.1,
) -> tuple[list[int], list[float], list[str]]:
    """
    Build per-step EMA performance for foraging runs.
    Performance is 'acc' if available, otherwise 1-loss.
    """
    steps_perf: List[int] = []
    perf_hist: List[float] = []
    task_hist: List[str] = []

    ema_per_task: Dict[str, float] = {}

    for r in rows:
        regime = r.get("regime", "")
        if regime != "foraging":
            continue

        mode = r.get("mode", "")
        if mode == "travel":
            continue

        step_str = r.get("step", "0")
        task = r.get("task", "")

        loss_str = r.get("loss", "")
        acc_str = r.get("acc", "")

        try:
            if acc_str not in ("", None):
                perf_raw = float(acc_str)
            elif loss_str not in ("", None):
                loss_val = float(loss_str)
                perf_raw = 1.0 - loss_val
            else:
                continue
        except ValueError:
            continue

        try:
            step = int(step_str)
        except ValueError:
            step = 0

        old = ema_per_task.get(task, perf_raw)
        new = (1.0 - alpha) * old + alpha * perf_raw
        ema_per_task[task] = new

        steps_perf.append(step)
        perf_hist.append(new)
        task_hist.append(task)

    return steps_perf, perf_hist, task_hist


def build_per_task_ema_dict(rows: List[Dict[str, Any]]) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    Construct generic per-task EMA histories from log rows.
    Returns:
        steps: sorted list of all steps encountered
        per_task: dict {task_name: [list of values aligned to steps, with NaNs for missing steps]}
    """
    all_steps = set()
    all_tasks = set()
    
    def _get_step(r):
        s = r.get("step", r.get("update"))
        return int(s) if s is not None else None

    # First pass: identify unique steps and tasks
    step_task_val = {} 

    for r in rows:
        s = _get_step(r)
        if s is None: continue
        
        t = r.get("task")
        if not t or t == "TRAVEL": continue
        
        val_str = r.get("ema")
        # fallback to acc if ema missing
        if val_str in (None, ""):
            val_str = r.get("acc")
            
        if val_str in (None, ""):
            continue
            
        try:
            val = float(val_str)
        except ValueError:
            continue
            
        all_steps.add(s)
        all_tasks.add(t)
        
        if s not in step_task_val:
            step_task_val[s] = {}
        step_task_val[s][t] = val

    if not all_steps:
        return [], {}

    sorted_steps = sorted(list(all_steps))
    sorted_tasks = sorted(list(all_tasks))
    
    per_task = {t: [] for t in sorted_tasks}
    
    # Forward fill logic
    last_vals = {t: np.nan for t in sorted_tasks}
    
    for s in sorted_steps:
        row_map = step_task_val.get(s, {})
        for t in sorted_tasks:
            if t in row_map:
                val = row_map[t]
                last_vals[t] = val
                per_task[t].append(val)
            else:
                per_task[t].append(last_vals[t])
                
    return sorted_steps, per_task
    return sorted_steps, per_task


def smooth_per_task_traces(per_task: Dict[str, List[float]], alpha: float) -> Dict[str, List[float]]:
    """
    Apply EMA smoothing to per-task traces.
    """
    if alpha >= 1.0:
        return per_task
        
    smoothed = {}
    for t, vals in per_task.items():
        # define min_periods=1 to avoid NaNs at start
        smoothed[t] = pd.Series(vals).ewm(alpha=alpha, adjust=False).mean().tolist()
    return smoothed

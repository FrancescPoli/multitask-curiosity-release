"""
Consolidated plotting functions for analysis.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
import torch

# NOTE: analysis.utils import moved inside functions to avoid circular dependency
# with curiosity package (which imports plots in __init__).

# ---------------------------------------------------------------------
# Generic training plots
# ---------------------------------------------------------------------

def _apply_axes_limits(
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Small helper to consistently set x/y limits if provided.
       Duplicated/local here or imported on demand to avoid cycles.
    """
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def plot_loss_and_ema(
    loss_hist: Sequence[float],
    acc_hist: Sequence[float],
    step_marks=None,
    title: str = "Training",
    xlim: Optional[Tuple[float, float]] = None,
    ylim_loss: Optional[Tuple[float, float]] = None,
    ylim_acc: Optional[Tuple[float, float]] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes[0].plot(loss_hist)
    axes[0].set_title(f"{title}: loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("CE")

    axes[1].plot(acc_hist)
    axes[1].set_title(f"{title}: accuracy EMA")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("Acc")

    if step_marks:
        for ax in axes:
            for s in step_marks:
                ax.axvline(s, linestyle="--", alpha=0.4)

    if xlim is not None:
        axes[0].set_xlim(xlim)
        axes[1].set_xlim(xlim)
    if ylim_loss is not None:
        axes[0].set_ylim(ylim_loss)
    if ylim_acc is not None:
        axes[1].set_ylim(ylim_acc)

    fig.tight_layout()
    return fig


def plot_per_task_ema_gapped(
    steps: List[int],
    per_task: Dict[str, List[float]],
    title: str = "Per-task EMA — NaN gaps",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    fig = plt.figure(figsize=(8, 4))
    for name, series in per_task.items():
        plt.plot(steps, series, label=name)
    plt.xlabel("update")
    plt.ylabel("EMA accuracy")
    plt.title(title)
    plt.legend()
    _apply_axes_limits(xlim, ylim)
    fig.tight_layout()
    return fig


def plot_per_task_bar(
    scores: Dict[str, float],
    title: str = "Final per-task evaluation",
):
    fig = plt.figure()
    names = list(scores.keys())
    vals = [scores[k] for k in names]
    plt.bar(range(len(names)), vals, tick_label=names)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(title)
    fig.tight_layout()
    return fig


def plot_p_arm(
    xs: List[int],
    ys: List[float],
    title: str = "Policy probability of chosen arm",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    fig = plt.figure(figsize=(10, 3))
    plt.plot(xs, ys, linestyle="--")
    plt.xlabel("update")
    plt.ylabel("p_arm")
    plt.title(title)
    _apply_axes_limits(xlim, ylim)
    fig.tight_layout()
    return fig


def plot_series(
    series: Sequence[float],
    title: str,
    xlabel: str = "update",
    ylabel: str = "value",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    fig = plt.figure(figsize=(8, 3))
    plt.plot(series)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _apply_axes_limits(xlim, ylim)
    fig.tight_layout()
    return fig


def plot_per_task_p_arm(
    rows: List[dict],
    title: str = "Per-task policy probabilities",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Plot p_arm over updates per task from policy logger rows."""
    updates = []
    by_task: Dict[str, List[tuple]] = {}
    for r in rows:
        step = int(r.get("step", r.get("update", 0)))
        task = str(r.get("task"))
        p = float(r.get("p_arm", r.get("prob", 0.0)))
        by_task.setdefault(task, []).append((step, p))

    fig = plt.figure(figsize=(10, 4))
    for task, series in by_task.items():
        series = sorted(series, key=lambda t: t[0])
        xs = [u for u, _ in series]
        ys = [p for _, p in series]
        plt.plot(xs, ys, label=task, linestyle="-")

    plt.xlabel("update")
    plt.ylabel("p(task chosen)")
    plt.title(title)
    if by_task:
        plt.legend(loc="best", fontsize="small", ncol=2)

    _apply_axes_limits(xlim, ylim)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Foraging plots
# ---------------------------------------------------------------------

def plot_foraging_rho_P(
    steps: List[int],
    rho_hist: List[float],
    P_hist: List[float],
    task_hist: List[str],
    tasks: List[str],
    title: str = "Foraging: global vs local reward",
    step_range: Optional[Tuple[int, int]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 3),
):
    """
    Plot global baseline rho(t) and local progress P(t) per task.
    """
    fig = plt.figure(figsize=figsize, dpi=150)

    steps_arr = np.array(steps)
    rho_arr = np.asarray(rho_hist)
    P_arr = np.asarray(P_hist)
    task_arr = np.asarray(task_hist, dtype=object)

    if step_range is not None:
        lo, hi = step_range
        mask = (steps_arr >= lo) & (steps_arr <= hi)
        steps_arr = steps_arr[mask]
        rho_arr = rho_arr[mask]
        P_arr = P_arr[mask]
        task_arr = task_arr[mask]

    cmap = plt.get_cmap("tab20")
    n_tasks = max(1, len(tasks))
    task_to_color = {t: cmap(i / max(1, n_tasks - 1)) for i, t in enumerate(tasks)}

    for t in tasks:
        mask_t = task_arr == t
        if not mask_t.any():
            continue
        y = np.full_like(P_arr, np.nan, dtype=float)
        y[mask_t] = P_arr[mask_t]
        plt.plot(steps_arr, y, color=task_to_color[t], linewidth=1.0, label=t)

    plt.plot(
        steps_arr,
        rho_arr,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="rho (global baseline)",
    )

    plt.xlabel("step")
    plt.ylabel("reward / learning progress")
    plt.title(title)

    if xlim is None and step_range is not None:
        xlim = (float(step_range[0]), float(step_range[1]))
    _apply_axes_limits(xlim, ylim)

    plt.tight_layout()
    return fig


def plot_foraging_decisions(
    steps: List[int],
    action_hist: List[int],
    title: str = "Foraging: stay / leave / travel over time",
    step_range: Optional[Tuple[int, int]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (8, 3),
):
    """
    Plot stay/leave/travel decisions over time.
    action_hist: -1 = travel, 0 = stay, 1 = leave
    """
    fig = plt.figure(figsize=figsize)
    x = np.array(steps)
    y = np.array(action_hist, dtype=float)

    if step_range is not None:
        lo, hi = step_range
        mask = (x >= lo) & (x <= hi)
        x = x[mask]
        y = y[mask]

    plt.scatter(x, y, s=8)
    plt.yticks([-1, 0, 1], ["travel", "stay", "leave"])
    plt.xlabel("step")
    plt.ylabel("decision")
    plt.title(title)

    if xlim is None and step_range is not None:
        xlim = (float(step_range[0]), float(step_range[1]))
    _apply_axes_limits(xlim, ylim)

    plt.tight_layout()
    return fig


def plot_foraging_per_task_perf(
    steps: List[int],
    perf_hist: List[float],
    task_hist: List[str],
    tasks: List[str],
    title: str = "Foraging: per-task performance (EMA)",
    step_range: Optional[Tuple[int, int]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 3),
):
    """
    Plot per-task performance (EMA) over time for foraging runs.
    """
    fig = plt.figure(figsize=figsize, dpi=150)

    steps_arr = np.array(steps)
    perf_arr = np.asarray(perf_hist, dtype=float)
    task_arr = np.asarray(task_hist, dtype=object)

    if step_range is not None:
        lo, hi = step_range
        mask = (steps_arr >= lo) & (steps_arr <= hi)
        steps_arr = steps_arr[mask]
        perf_arr = perf_arr[mask]
        task_arr = task_arr[mask]

    cmap = plt.get_cmap("tab20")
    n_tasks = max(1, len(tasks))
    task_to_color = {t: cmap(i / max(1, n_tasks - 1)) for i, t in enumerate(tasks)}

    for t in tasks:
        mask_t = task_arr == t
        if not mask_t.any():
            continue
        y = np.full_like(perf_arr, np.nan, dtype=float)
        y[mask_t] = perf_arr[mask_t]
        plt.plot(
            steps_arr,
            y,
            color=task_to_color[t],
            linewidth=1.0,
            label=t,
        )

    plt.xlabel("step")
    plt.ylabel("performance (EMA)")
    plt.title(title)
    if tasks:
        plt.legend(fontsize=7, ncol=2)

    if xlim is None and step_range is not None:
        xlim = (float(step_range[0]), float(step_range[1]))
    _apply_axes_limits(xlim, ylim)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Model Weights (Wrec) plots
# ---------------------------------------------------------------------

cmap_div = LinearSegmentedColormap.from_list(
    "rwg", ["#7f0000", "#ffffff", "#00441b"], N=256
)  # neg → white → pos
cmap_pos = LinearSegmentedColormap.from_list(
    "wg", ["#ffffff", "#00441b"], N=256
)  # 0 → pos
cmap_neg = LinearSegmentedColormap.from_list(
    "rw", ["#7f0000", "#ffffff"], N=256
)  # neg → 0


def show_heatmap(A, title: str, xlab: str, ylab: str):
    """
    Generic heatmap with diverging/one-sided colormaps depending on sign of values.
    """
    A = A.detach().cpu().numpy() if hasattr(A, "detach") else np.asarray(A)
    if A.ndim == 1:
        A = A[None, :]  # show 1D as single row

    amin, amax = float(A.min()), float(A.max())

    if amin < 0.0 and amax > 0.0:
        norm = TwoSlopeNorm(vmin=amin, vcenter=0.0, vmax=amax)
        cmap = cmap_div
    elif amax <= 0.0:
        norm = Normalize(vmin=amin, vmax=0.0)
        cmap = cmap_neg
    else:
        norm = Normalize(vmin=0.0, vmax=amax)
        cmap = cmap_pos

    plt.figure(figsize=(8, 6))
    im = plt.imshow(A, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Value")
    plt.tight_layout()
    cbar.set_label("Value")
    plt.tight_layout()
    return plt.gcf()


def plot_Wrec_and_hist(W_rec: torch.Tensor):
    """
    Plot heatmap and histogram of recurrent weights.
    """
    fig1 = show_heatmap(W_rec, "LeakyRNN hidden→hidden weights (W_rec)",
                 "From hidden (t-1)", "To hidden (t)")

    w = W_rec.detach().cpu().numpy().ravel()
    w_nz = w[w != 0]

    fig2 = plt.figure(figsize=(8, 5))
    plt.hist(w_nz, bins="auto")
    plt.title("Histogram of non-zero hidden→hidden weights (W_rec)")
    plt.xlabel("Weight value")
    plt.ylabel("Count")
    plt.tight_layout()
    
    print(f"[stats] non-zero W_rec entries: {w_nz.size} / total: {w.size}")
    return [fig1, fig2]


def plot_brain_mask(mask: np.ndarray):
    """Plot a brain mask with the same heatmap helper and print sparsity."""
    fig = show_heatmap(mask, "Brain mask", "Columns", "Rows")
    nnz = int((mask != 0).sum())
    print(f"[mask] non-zero elements in mask: {nnz} / total: {mask.size}")
    return fig


def plot_Wrec_density_over_checkpoints(
    run_dir: Path,
    meta: dict,
    device: torch.device = torch.device("cpu"),
    nbins: int = 80,
) -> None:
    """
    Load each intermediate checkpoint, extract W_rec, and plot the
    density (histogram) of recurrent weights across training.
    """
    from analysis.plotting.utils import (
        get_Wrec,
        list_checkpoint_steps_and_paths,
        build_model_from_meta,
        load_state_into_model
    )
    
    ckpts = list_checkpoint_steps_and_paths(run_dir)
    if not ckpts:
        print("No intermediate checkpoints (state_step*.pt) found; skipping weight-density plot.")
        return

    print(f"Found {len(ckpts)} intermediate checkpoints.")

    # Build a model once, then reuse it for all checkpoints
    model = build_model_from_meta(meta, device=device)

    all_w_nz: list[np.ndarray] = []
    steps: list[int] = []

    for step, path in ckpts:
        obj = torch.load(path, map_location=device)
        if isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        else:
            state = obj

        # Load weights
        load_state_into_model(model, state)

        # Extract recurrent weights
        W_rec = get_Wrec(model)
        w = W_rec.detach().cpu().numpy().ravel()
        w_nz = w[w != 0]

        if w_nz.size == 0:
            continue

        all_w_nz.append(w_nz)
        steps.append(step)

    if not all_w_nz:
        print("Checkpoints found, but W_rec appears to be all zeros; skipping density plot.")
        return

    # Common binning across all checkpoints
    global_min = min(w.min() for w in all_w_nz)
    global_max = max(w.max() for w in all_w_nz)
    if global_min == global_max:
        global_min -= 1e-6
        global_max += 1e-6

    edges = np.linspace(global_min, global_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    dens = np.zeros((len(all_w_nz), nbins), dtype=float)
    for i, w in enumerate(all_w_nz):
        hist, _ = np.histogram(w, bins=edges, density=True)
        dens[i, :] = hist

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        dens,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(im, label="Density")

    xticks_idx = np.linspace(0, nbins - 1, 5, dtype=int)
    plt.xticks(
        xticks_idx,
        [f"{centers[i]:.2f}" for i in xticks_idx],
    )
    plt.xlabel("W_rec weight value")

    n_ckpts = len(steps)
    if n_ckpts <= 10:
        yticks_idx = np.arange(n_ckpts)
    else:
        yticks_idx = np.linspace(0, n_ckpts - 1, 10, dtype=int)

    plt.yticks(
        yticks_idx,
        [str(steps[i]) for i in yticks_idx],
    )
    plt.ylabel("Checkpoint step")

    plt.title("Density of W_rec weights across checkpoints")
    plt.tight_layout()
    plt.title("Density of W_rec weights across checkpoints")
    plt.tight_layout()
    return plt.gcf()


def plot_Wrec_L1_over_checkpoints(
    run_dir: Path,
    meta: dict,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    For each intermediate checkpoint, compute and plot L1 norm of W_rec.
    """
    from analysis.plotting.utils import (
        get_Wrec,
        list_checkpoint_steps_and_paths,
        build_model_from_meta,
        load_state_into_model
    )
    
    ckpts = list_checkpoint_steps_and_paths(run_dir)
    if not ckpts:
        print("No intermediate checkpoints (state_step*.pt) found; skipping W_rec L1 plot.")
        return

    print(f"Computing W_rec L1 norm for {len(ckpts)} checkpoints...")

    model = build_model_from_meta(meta, device=device)

    steps: list[int] = []
    l1_values: list[float] = []

    for step, path in ckpts:
        obj = torch.load(path, map_location=device)
        if isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        else:
            state = obj

        load_state_into_model(model, state)
        W_rec = get_Wrec(model)
        l1 = W_rec.detach().abs().sum().item()

        steps.append(step)
        l1_values.append(l1)

    if not steps:
        print("No usable checkpoints for W_rec L1 plot; skipping.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(steps, l1_values, marker="o")
    plt.xlabel("checkpoint step")
    plt.ylabel("∑ |W_rec|")
    plt.title("L1 norm of W_rec across checkpoints")
    plt.tight_layout()
    plt.title("L1 norm of W_rec across checkpoints")
    plt.tight_layout()
    return plt.gcf()


# ---------------------------------------------------------------------
# Scaffolding / Analysis Plots
# ---------------------------------------------------------------------

def plot_scaffolding_task_selection(df: pd.DataFrame, step_range=None):
    """
    Plot Task Selection Over Time.
    """
    df_tasks = df[(df['regime'] == 'foraging') & (df['mode'] != 'travel')].copy()
    
    if step_range:
        df_tasks = df_tasks[(df_tasks['step'] >= step_range[0]) & (df_tasks['step'] <= step_range[1])]
    
    tasks = sorted(df_tasks['task'].unique())
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    df_tasks['task_idx'] = df_tasks['task'].map(task_to_idx)
    
    plt.figure(figsize=(14, 6), dpi=120)
    colors = plt.cm.tab20(np.linspace(0, 1, len(tasks)))
    
    for i, task in enumerate(tasks):
        subset = df_tasks[df_tasks['task'] == task]
        plt.scatter(subset['step'], subset['task_idx'], s=3, alpha=0.6, 
                   color=colors[i], label=task if len(tasks) <= 20 else None)
    
    plt.yticks(range(len(tasks)), [t.split('.')[-1] for t in tasks], fontsize=8)
    plt.xlabel("Training Step", fontsize=10)
    plt.ylabel("Task", fontsize=10)
    plt.title("Task Selection Over Time (Scaffolding + Foraging)", fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return plt.gcf()


def plot_scaffolding_first_appearance(df: pd.DataFrame):
    """
    Plot First Appearance of Each Task.
    """
    df_tasks = df[(df['regime'] == 'foraging') & (df['mode'] != 'travel')].copy()
    if df_tasks.empty:
        print("No foraging tasks found in DataFrame.")
        return 
        
    first_seen = df_tasks.groupby('task')['step'].min().sort_values()
    
    print("\n📊 First appearance of tasks (reveals scaffolding groups):")
    for task, step in first_seen.items():
        print(f"  {step:6d}  {task}")
    
    plt.figure(figsize=(12, 5), dpi=120)
    colors = plt.cm.viridis(np.linspace(0, 1, len(first_seen)))
    
    plt.bar(range(len(first_seen)), first_seen.values, color=colors)
    plt.xticks(range(len(first_seen)), 
               [t.split('.')[-1] for t in first_seen.index], 
               rotation=45, ha='right', fontsize=8)
    plt.ylabel("First Appearance Step", fontsize=10)
    plt.title("Scaffolding Schedule: First Appearance of Each Task", fontsize=12)
    
    steps_sorted = sorted(first_seen.values)
    for i in range(1, len(steps_sorted)):
        if steps_sorted[i] - steps_sorted[i-1] > 500:
            plt.axhline(y=(steps_sorted[i] + steps_sorted[i-1])/2, 
                       color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.tight_layout()
    return plt.gcf()


def plot_task_switching_analysis(df: pd.DataFrame, step_range=None):
    """
    Plot Task Switching Analysis (switches and block durations).
    """
    df_train = df[(df['regime'] == 'foraging') & (df['mode'] != 'travel')].copy()
    
    if step_range:
        df_train = df_train[(df_train['step'] >= step_range[0]) & (df_train['step'] <= step_range[1])]
    
    if df_train.empty:
         print("No foraging data found for switching analysis.")
         return

    # Count switches
    df_train['task_changed'] = df_train['task'] != df_train['task'].shift(1)
    switches = df_train[df_train['task_changed']].copy()
    
    print(f"\n📊 Task Switching Statistics:")
    print(f"  Total training steps: {len(df_train)}")
    print(f"  Total task switches: {len(switches)}")
    if len(switches) > 0:
        print(f"  Average steps between switches: {len(df_train) / max(1, len(switches)):.1f}")
    
    # Block durations
    df_train['block_id'] = df_train['task_changed'].cumsum()
    block_lengths = df_train.groupby('block_id').size()
    
    print(f"  Median block length: {block_lengths.median():.0f} steps")
    print(f"  Mean block length: {block_lengths.mean():.1f} steps")
    print(f"  Min block length: {block_lengths.min()} steps")
    print(f"  Max block length: {block_lengths.max()} steps")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
    
    # Histogram
    axes[0].hist(block_lengths, bins=30, edgecolor='white', alpha=0.7)
    axes[0].axvline(block_lengths.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {block_lengths.median():.0f}')
    axes[0].set_xlabel("Block Length (steps on task)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Block Lengths")
    axes[0].legend()
    
    # Block lengths over time
    block_starts = df_train.groupby('block_id')['step'].min()
    axes[1].scatter(block_starts, block_lengths, s=10, alpha=0.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Block Length")
    axes[1].set_title("Block Lengths Over Training Time")
    
    plt.tight_layout()
    plt.tight_layout()
    return plt.gcf(), block_lengths

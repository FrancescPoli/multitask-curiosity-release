"""
Publication-quality task-visualisation panels.

Based DIRECTLY on visualize_all_tasks.py — target overlay logic copied verbatim.
Uses dataset_mode='stream' (like the original) so seq_len drives how many trials appear.

Layout: N tasks as portrait columns (height ≈ 4× width), N_TRIALS complete trials each.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from curiosity.data import make_dataset

# Light-blue colour for the response-target overlay (fits viridis palette)
RESP_RGBA = [1.0, 1.0, 1.0, 1.0]

# ── Config ─────────────────────────────────────────────────────────────────────
TASKS_TO_SHOW = [
    "poli.go",
    "poli.dlygo",
    "poli.antigo",
    "poli.ctxgo",
    "poli.dm1",
    "poli.dlyms",
]

TASK_LABELS = {
    "poli.go":      "Go",
    "poli.dlygo":   "Delay-Go",
    "poli.antigo":  "Anti-Go",
    "poli.ctxgo":   "Context-Go",
    "poli.dm1":     "Decision-\nmaking",
    "poli.dlyms":   "Match-to-\nSample",
}

# Fixed timing overrides for cleaner visualization
# dm1: balanced fixation vs stimulus vs decision
# dlyms: equal sample and test durations
TASK_TIMING = {
    "poli.dm1":   {'fixation': 800, 'stimulus': 400, 'decision': 500},
    "poli.dlyms": {'fixation': 400, 'sample': 500, 'delay': 800, 'test': 500, 'decision': 500},
}

# seq_len: with fixed timing overrides all trials are predictable; 600 is enough for all
TASK_SEQ_LEN = {}
DEFAULT_SEQ_LEN = 600

# How many complete trials to show per task
TASK_N_TRIALS = {}
DEFAULT_N_TRIALS = 1

DT       = 20
E_THRESH = 0.1

# ── Trial cutter ───────────────────────────────────────────────────────────────
def get_n_trials(ob: np.ndarray, gt: np.ndarray, n: int):
    """Return (ob, gt) trimmed to exactly n complete trials.
    Stream mode: fixation goes 0→1 at the START of each new trial.
    Smooth fixation first to suppress Gaussian noise added during stimulus period.
    """
    fix        = ob[:, 0]
    fix_smooth = np.convolve(fix, np.ones(7) / 7, mode='same')
    onsets     = np.where((fix_smooth[:-1] < 0.5) & (fix_smooth[1:] >= 0.5))[0] + 1
    if len(onsets) >= 2:
        start = onsets[0]
        cut   = onsets[n] if len(onsets) > n else ob.shape[0]
        return ob[start:cut], gt[start:cut]
    # fallback: return as-is
    return ob, gt

# ── Panel draw — logic copied verbatim from visualize_all_tasks.py ─────────────
def draw_task_panel(ax, ob, gt, task_name, title, show_ylabel):
    time_dim  = ob.shape[0]
    input_dim = ob.shape[1]

    # ── Active ring mask ─────────────────────────────────────────────────────
    mask_r1  = np.zeros(time_dim, dtype=bool)
    mask_r2  = np.zeros(time_dim, dtype=bool)
    state_r1 = True
    state_r2 = False
    has_r2   = (input_dim >= 33)

    for t in range(time_dim):
        e1 = np.mean(ob[t, 1:17])
        e2 = np.mean(ob[t, 17:33]) if has_r2 else 0.0
        is_on_1 = e1 > E_THRESH
        is_on_2 = e2 > E_THRESH
        if is_on_1 and is_on_2:
            state_r1 = True;  state_r2 = True
        elif is_on_1:
            state_r1 = True;  state_r2 = False
        elif is_on_2:
            state_r1 = False; state_r2 = True
        mask_r1[t] = state_r1
        mask_r2[t] = state_r2

    # ── Background ───────────────────────────────────────────────────────────
    ax.imshow(ob.T, aspect='auto', interpolation='nearest',
              origin='lower', cmap='viridis', alpha=0.9)

    # ── Target overlay — force_ring logic verbatim from visualize_all_tasks.py
    target_rgba = np.zeros((input_dim, time_dim, 4))

    force_ring = None
    force_both = False
    if 'multi' in task_name:
        force_both = True
    elif 'dm2' in task_name or 'catdm2' in task_name:
        force_ring = 16
    elif 'dm1' in task_name or 'catdm1' in task_name:
        force_ring = 0
    elif 'ctx' in task_name or 'context' in task_name:
        force_ring = 0

    for t in range(time_dim):
        cls = gt[t]
        if cls > 0:
            idx0 = int(cls)
            idx1 = int(cls + 16)

            if force_both:
                if 0 <= idx0 < input_dim:
                    target_rgba[idx0, t, :] = RESP_RGBA
                if has_r2 and 0 <= idx1 < input_dim:
                    target_rgba[idx1, t, :] = RESP_RGBA
                continue

            if force_ring is not None:
                idx_forced = int(cls + force_ring)
                if 0 <= idx_forced < input_dim:
                    target_rgba[idx_forced, t, :] = RESP_RGBA
                continue

            # Default hybrid logic
            val0 = ob[t, idx0] if idx0 < input_dim else 0.0
            val1 = ob[t, idx1] if (has_r2 and idx1 < input_dim) else 0.0
            has_signal0 = val0 > E_THRESH
            has_signal1 = val1 > E_THRESH

            if has_signal0 or has_signal1:
                if has_signal0:
                    target_rgba[idx0, t, :] = RESP_RGBA
                if has_signal1:
                    target_rgba[idx1, t, :] = RESP_RGBA
            else:
                if mask_r1[t] and 0 <= idx0 < input_dim:
                    target_rgba[idx0, t, :] = RESP_RGBA
                if mask_r2[t] and has_r2 and 0 <= idx1 < input_dim:
                    target_rgba[idx1, t, :] = RESP_RGBA

    ax.imshow(target_rgba, aspect='auto', interpolation='nearest', origin='lower')

    # ── Y-ticks ──────────────────────────────────────────────────────────────
    ticks_r1 = np.arange(1, 17)
    ticks_r2 = np.arange(17, 33)
    all_ticks  = np.concatenate([ticks_r1, ticks_r2])
    all_labels = [str(l) for l in list(range(1, 17)) + list(range(1, 17))]
    valid        = all_ticks < input_dim
    final_ticks  = all_ticks[valid]
    final_labels = np.array(all_labels)[valid]

    # sparse ticks for publication — labels only on leftmost panel
    step = 1
    ax.set_yticks(final_ticks[::step])
    ax.set_yticklabels(final_labels[::step] if show_ylabel else [], fontsize=8)

    ax.axhline(y=0.5,  color='white', linewidth=0.5, alpha=0.4)
    ax.axhline(y=16.5, color='white', linewidth=0.5, alpha=0.6)

    ax.set_xticks([])
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    if show_ylabel:
        ax.set_ylabel("Input channel", fontsize=9, labelpad=3)
    ax.tick_params(axis='y', length=2, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

# ── Main ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

N           = len(TASKS_TO_SHOW)
PANEL_H     = 4.5
LEGEND_H    = 0.55   # height of legend strip in inches
# Per-task column width multipliers (go tasks narrow, DM/dlyms wider)
WIDTH_RATIOS = [1.0, 1.0, 1.0, 1.0, 1.9, 1.9]
UNIT_W      = 0.8   # base width for a ratio-1 column, in inches
total_w     = sum(r * UNIT_W for r in WIDTH_RATIOS)
fig_h       = PANEL_H + 0.6 + LEGEND_H + 0.15
fig = plt.figure(figsize=(total_w + 0.7, fig_h))
# panels sit above the legend strip
bottom_frac = (LEGEND_H + 0.2) / fig_h
gs  = gridspec.GridSpec(1, N, figure=fig,
                        left=0.09, right=0.99,
                        top=0.97, bottom=bottom_frac,
                        wspace=0.15,
                        width_ratios=WIDTH_RATIOS)

for col, task in enumerate(TASKS_TO_SHOW):
    seq_len  = TASK_SEQ_LEN.get(task, DEFAULT_SEQ_LEN)
    n_trials = TASK_N_TRIALS.get(task, DEFAULT_N_TRIALS)
    print(f"  {task} (seq_len={seq_len}, n_trials={n_trials}) ...", end=" ", flush=True)
    try:
        ds = make_dataset(task, dt=DT, batch_size=1, seq_len=seq_len,
                          dataset_mode='stream')
        # Override timing after env creation (poli envs don't accept timing as kwarg)
        if task in TASK_TIMING:
            inner = ds.env.unwrapped if hasattr(ds.env, 'unwrapped') else ds.env
            inner.timing.update(TASK_TIMING[task])
        inp, tgt = ds()
        ob  = inp[:, 0, :]
        gt  = tgt[:, 0]
        ob, gt = get_n_trials(ob, gt, n_trials)

        ax = fig.add_subplot(gs[0, col])
        draw_task_panel(ax, ob, gt, task,
                        title=TASK_LABELS.get(task, task),
                        show_ylabel=(col == 0))
        print(f"ok (T={ob.shape[0]})")
    except Exception as e:
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                ha="center", va="center", fontsize=6, color="red", wrap=True)
        ax.axis("off")
        print(f"FAILED: {e}")

# ── Legend strip ──────────────────────────────────────────────────────────────
leg_h_frac  = LEGEND_H / fig_h
leg_y0      = 0.01 / fig_h          # small bottom margin in fig coords
leg_y1      = leg_y0 + leg_h_frac

# Left: patch legend (fixation + response)
patch_ax = fig.add_axes([0.05, leg_y0, 0.38, leg_h_frac])
patch_ax.set_axis_off()
fix_color  = mcm.get_cmap("viridis")(1.0)   # yellow — same as fix=1 in heatmap
resp_color = tuple(RESP_RGBA[:3])
patches = [
    mpatches.Patch(facecolor=fix_color,  edgecolor="#555", linewidth=0.5,
                   label="Fixation period"),
    mpatches.Patch(facecolor=resp_color, edgecolor="#555", linewidth=0.5,
                   label="Response target"),
]
patch_ax.legend(handles=patches, loc="center", ncol=2, fontsize=9,
                frameon=False, handlelength=1.6, handleheight=1.3,
                columnspacing=1.2)

# Right: colorbar for stimulus intensity (blue → light green = lower viridis range)
cbar_ax = fig.add_axes([0.52, leg_y0 + leg_h_frac * 0.25,
                        0.42, leg_h_frac * 0.45])
viridis     = mcm.get_cmap("viridis")
stim_colors = viridis(np.linspace(0.0, 0.62, 256))   # blue → green (below yellow)
stim_cmap   = mcolors.LinearSegmentedColormap.from_list("stim", stim_colors)
sm = mcm.ScalarMappable(cmap=stim_cmap, norm=mcolors.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Stimulus intensity", fontsize=9, labelpad=3)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Low", "High"], fontsize=8)
cbar.outline.set_linewidth(0.5)

OUT = os.path.join(os.path.dirname(__file__), "Figures", "pub_task_examples")
plt.savefig(OUT + ".pdf", dpi=300, bbox_inches="tight")
plt.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {OUT}.pdf / .png")

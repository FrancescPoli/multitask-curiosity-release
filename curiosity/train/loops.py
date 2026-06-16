from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import time
import torch
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..data import sample_batch_with_decision
from ..utils.value_functions import make_value_function
from ..metrics import (
    batch_decision_accuracy_seqwise,
    ce_loss_on_decision_frames,
    evaluate_task,
    ce_all_nonpadding_frames,
    ce_yang_cmask,  # Yang-style temporal weighting
    activity_reg,
)
from ..logging_io import write_csv_rows
from ..checkpoints import save_state_dict
from .foraging import MVTCurriculumController


@dataclass
class TrainResult:
    loss_hist: List[float]
    acc_hist: List[float]
    ema_hist: Dict[str, List[float]]
    steps_hist: List[int]
    eval_final: Dict[str, float]
    artifacts: Dict[str, Path]

# ---------------- L1 regularization helpers ----------------

def _select_weight_params(model: torch.nn.Module, mode: str):
    """
    Yield tensors to regularize according to mode.
    modes:
      - 'all' -> all params
      - 'no-bias' -> params with dim>=2 (skip biases/1D)
      - 'recurrent-only' -> only recurrent block (Wrec or bottom block of combined W)
    """
    mode = (mode or 'no-bias').lower()
    if mode == 'recurrent-only':
        if hasattr(model, 'cell_sep') and getattr(model, 'cell_sep') is not None:
            yield model.cell_sep.Wrec
        elif hasattr(model, 'cell') and getattr(model, 'cell') is not None:
            p = model.cell.W  # shape [n_in + n_hid, n_hid]
            n_in = int(model.hp.n_input)
            yield p[n_in:, :]
        return
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if mode == 'no-bias' and p.dim() == 1:
            continue
        yield p

def l1_weight_penalty(model: torch.nn.Module, lam: float, mode: str = 'no-bias') -> torch.Tensor:
    if lam <= 0:
        # create a zero tensor on the right device to avoid dtype/device mismatch
        for p in model.parameters():
            return p.sum() * 0.0
        return torch.tensor(0.0)
    reg = None
    for p in _select_weight_params(model, mode):
        term = p.abs().sum()
        reg = term if reg is None else (reg + term)
    if reg is None:
        for p in model.parameters():
            return p.sum() * 0.0
        return torch.tensor(0.0)
    return lam * reg

@torch.no_grad()
def apply_prox_l1_(model: torch.nn.Module, lam: float, step_size: float, mode: str = 'no-bias') -> None:
    """Soft-threshold in-place: w <- sign(w) * max(|w| - eta*lam, 0)."""
    if lam <= 0 or step_size <= 0:
        return
    thresh = float(step_size * lam)
    mode = (mode or 'no-bias').lower()
    if mode == 'recurrent-only':
        if hasattr(model, 'cell_sep') and getattr(model, 'cell_sep') is not None:
            p = model.cell_sep.Wrec
            p.copy_(torch.sign(p) * torch.clamp(p.abs() - thresh, min=0.0))
        elif hasattr(model, 'cell') and getattr(model, 'cell') is not None:
            p = model.cell.W
            n_in = int(model.hp.n_input)
            sl = p[n_in:, :]
            sl.copy_(torch.sign(sl) * torch.clamp(sl.abs() - thresh, min=0.0))
        return
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if mode == 'no-bias' and p.dim() == 1:
            continue
        p.copy_(torch.sign(p) * torch.clamp(p.abs() - thresh, min=0.0))

def _get_Wrec(model: torch.nn.Module) -> Optional[torch.Tensor]:
    if hasattr(model, 'cell_sep') and getattr(model, 'cell_sep') is not None:
        return model.cell_sep.Wrec
    if hasattr(model, 'cell') and getattr(model, 'cell') is not None:
        p = model.cell.W
        n_in = int(model.hp.n_input)
        return p[n_in:, :]
    return None

def distance_penalty(model: torch.nn.Module, lam: float, power: float = 1.0) -> torch.Tensor:
    """
    λ · sum( |Wrec| ⊙ D^power ), where D is an H×H distance matrix
    stored on model.distance_matrix. Returns a proper zero tensor if
    lam<=0 or D/Wrec unavailable.
    """
    if lam <= 0:
        for p in model.parameters():
            return p.sum() * 0.0
        return torch.tensor(0.0)

    Wrec = _get_Wrec(model)
    D = getattr(model, "distance_matrix", None)
    if Wrec is None or D is None:
        for p in model.parameters():
            return p.sum() * 0.0
        return torch.tensor(0.0)

    D = D.to(Wrec.device, dtype=Wrec.dtype)
    if power != 1.0:
        D = torch.pow(D, power)
    return lam * torch.sum(torch.abs(Wrec) * D)

# ---------------- training utilities ----------------

def _maybe_print(step: int, kv: Dict[str, str], every: int) -> None:
    if every > 0 and (step % every == 0 or step == 1):
        msg = " | ".join([f"{k}:{v}" for k, v in kv.items()])
        print(f"[{step:6d}] {msg}")

def _maybe_save_intermediate(
    cfg,
    model: torch.nn.Module,
    run_dir: Path,
    step: int,
    ckpts: Optional[deque] = None,
) -> None:
    """
    Optionally save an intermediate state_dict at this training step.

    Controlled by:
        cfg.save_every:      if <= 0, disabled; otherwise every N steps.
        cfg.max_checkpoints: if > 0, keep at most this many (rolling).
    """
    save_every = getattr(cfg, "save_every", 0) or 0
    if save_every <= 0:
        return
    if step % save_every != 0:
        return

    path = run_dir / f"state_step{step:06d}.pt"
    save_state_dict(model, path)

    max_ckpt = getattr(cfg, "max_checkpoints", 0) or 0
    if ckpts is not None and max_ckpt > 0:
        ckpts.append(path)
        if len(ckpts) > max_ckpt:
            old = ckpts.popleft()
            try:
                old.unlink()
            except OSError:
                pass


# ---------------- training loops ----------------

def train_joint(
    cfg,
    datasets: Dict[str, Any],
    model: torch.nn.Module,
    opt,
    device,
    run_dir: Path,
    scaffolder=None,
) -> TrainResult:
    env0 = next(iter(datasets.values())).env
    n_actions = env0.action_space.n

    loss_hist: List[float] = []
    acc_hist: List[float] = []
    steps_hist: List[int] = []
    ema_hist = {t: [] for t in cfg.tasks}
    base_ema_per_task = {t: None for t in cfg.tasks}
    csv_rows: List[Dict[str, Any]] = []

    start = time.time()
    ckpts = deque()

    alpha_ema = getattr(cfg, "pg_ema_alpha", 0.9)
    
    # Random number generator for task sampling (Yang-style)
    rng = np.random.RandomState(cfg.seed)

    for step in range(1, cfg.steps + 1):
        # Determine active tasks
        if scaffolder is not None:
            active_list = scaffolder.get_active_tasks(step)
        else:
            active_list = list(datasets.keys())

        # CRITICAL FIX: Sample ONE random task per step (Yang's approach)
        # This prevents catastrophic forgetting from sequential updates
        task_name = rng.choice(active_list)
        
        if task_name not in datasets:
            continue
        ds = datasets[task_name]
        
        sigma_x = getattr(cfg, 'sigma_x', 0.0)
        alpha = getattr(cfg, 'dt', 20.0) / 100.0
        x, y_labels, T, B = sample_batch_with_decision(ds, device=device, sigma_x=sigma_x, alpha=alpha)
        opt.zero_grad()
        outs = model(x)
        logits = outs["ring_logits"]
        h = outs["h"]

        loss = ce_yang_cmask(logits, y_labels, decision_weight=5.0)  # Yang's c_mask weighting

        # Regularisers (unchanged)
        if getattr(model.hp, "l1_h", 0.0) > 0 or getattr(model.hp, "l2_h", 0.0) > 0:
            loss = loss + activity_reg(
                h,
                getattr(model.hp, "l1_h", 0.0),
                getattr(model.hp, "l2_h", 0.0),
            )

        if getattr(cfg, "l1_weight", 0.0) > 0:
            loss = loss + l1_weight_penalty(
                model, cfg.l1_weight, getattr(cfg, "l1_on", "no-bias")
            )

        if getattr(cfg, "distance_penalty", False) and getattr(cfg, "distance_weight", 0.0) > 0:
            loss = loss + distance_penalty(
                model,
                cfg.distance_weight,
                getattr(cfg, "distance_power", 1.0),
            )

        loss.backward()
        if getattr(cfg, 'grad_clip_mode', 'norm') == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if getattr(cfg, "prox_l1_weight", 0.0) > 0:
            apply_prox_l1_(
                model,
                cfg.prox_l1_weight,
                cfg.lr,
                getattr(cfg, "prox_l1_on", "no-bias"),
            )

        if hasattr(model, "reapply_brain_constraints_"):
            model.reapply_brain_constraints_()

            with torch.no_grad():
                _, logits2d, _ = ce_loss_on_decision_frames(logits, y_labels, n_actions)
                acc = batch_decision_accuracy_seqwise(
                    logits2d, y_labels.reshape(-1), T, B
                )

                loss_hist.append(loss.item())
                acc_hist.append(float(acc))
                steps_hist.append(step)

                # EMA per task
                ema_prev = base_ema_per_task[task_name]
                new_ema = (
                    alpha_ema * ema_prev + (1 - alpha_ema) * float(acc)
                    if ema_prev is not None
                    else float(acc)
                )
                base_ema_per_task[task_name] = new_ema
                for tn in cfg.tasks:
                    ema_hist[tn].append(
                        base_ema_per_task[tn]
                        if base_ema_per_task[tn] is not None
                        else 0.0
                    )

                # Rich CSV row (per task, per step)
                csv_rows.append(
                    {
                        "step": int(step),
                        "mode": "joint",
                        "task": task_name,
                        "loss": float(loss.item()),
                        "acc": float(acc),
                        "ema": float(new_ema),
                    }
                )

        # Logging 
        _maybe_print(
            step,
            {
                "mode": "joint",
                "loss": f"{loss_hist[-1]:.4f}",
                "acc": f"{acc_hist[-1]:.3f}",
                "time_s": f"{time.time() - start:.1f}",
            },
            cfg.log_every,
        )

        # optional intermediate checkpoint
        _maybe_save_intermediate(cfg, model, run_dir, step, ckpts)

    # Final eval (unchanged)
    eval_results = {}
    with torch.no_grad():
        model.eval()
        for tn, ds in datasets.items():
            acc = evaluate_task(model, ds.env, device=device, ntrial=max(50, cfg.steps // 20),
                                rule_idx=ds.rule_idx, n_rules=ds.n_rules)
            eval_results[tn] = float(acc)

    # Always write curriculum CSV if we collected rows
    artifacts = {}
    if csv_rows:
        csv_path = run_dir / "run_curriculum.csv"
        write_csv_rows(csv_path, csv_rows)
        artifacts["curriculum_rows"] = csv_path

    return TrainResult(loss_hist, acc_hist, ema_hist, steps_hist, eval_results, artifacts=artifacts)


def train_sequential(cfg, datasets: Dict[str, Any], model: torch.nn.Module, opt, device, run_dir: Path) -> TrainResult:
    env0 = next(iter(datasets.values())).env
    n_actions = env0.action_space.n

    loss_hist: List[float] = []
    acc_hist: List[float] = []
    steps_hist: List[int] = []
    ema_hist = {t: [] for t in cfg.tasks}
    csv_rows: List[Dict[str, Any]] = []
    start = time.time()
    ckpts = deque()

    step = 0
    for task_name, ds in datasets.items():
        for _ in range(cfg.steps_per_block):
            step += 1
            sigma_x = getattr(cfg, 'sigma_x', 0.0)
            alpha = getattr(cfg, 'dt', 20.0) / 100.0
            x, y_labels, T, B = sample_batch_with_decision(ds, device=device, sigma_x=sigma_x, alpha=alpha)
            opt.zero_grad()
            outs = model(x)
            logits = outs["ring_logits"]
            h_seq = outs["h"]

            # One-head CE on all non-padding frames (robust to any negative padding)
            loss = ce_yang_cmask(logits, y_labels, decision_weight=5.0)  # Yang's c_mask weighting

            # Decision-only metric (unchanged behaviour)
            _, logits2d, _ = ce_loss_on_decision_frames(logits, y_labels, n_actions)

            # Optional hidden-activity regularization
            if getattr(model.hp, "l1_h", 0.0) > 0 or getattr(model.hp, "l2_h", 0.0) > 0:
                loss = loss + activity_reg(h_seq, getattr(model.hp, "l1_h", 0.0), getattr(model.hp, "l2_h", 0.0))

            # Optional weights regularization
            if getattr(cfg, 'l1_weight', 0.0) > 0:
                loss = loss + l1_weight_penalty(model, cfg.l1_weight, getattr(cfg, 'l1_on', 'no-bias'))
            
            # Optional distance regularization
            if getattr(cfg, 'distance_penalty', False) and getattr(cfg, 'distance_weight', 0.0) > 0:
                loss = loss + distance_penalty(model, cfg.distance_weight, getattr(cfg, 'distance_power', 1.0))

            loss.backward()
            if getattr(cfg, 'grad_clip_mode', 'norm') == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if getattr(cfg, 'prox_l1_weight', 0.0) > 0:
                apply_prox_l1_(model, cfg.prox_l1_weight, cfg.lr, getattr(cfg, 'prox_l1_on', 'no-bias'))
            if hasattr(model, 'reapply_brain_constraints_'):
                model.reapply_brain_constraints_()

            with torch.no_grad():
                acc = batch_decision_accuracy_seqwise(logits2d, y_labels.reshape(-1), T, B)
                loss_hist.append(loss.item())
                acc_hist.append(float(acc))
                steps_hist.append(step)
                for tn in cfg.tasks:
                    ema_hist[tn].append(float(acc) if tn == task_name else (ema_hist[tn][-1] if ema_hist[tn] else 0.0))
                csv_rows.append({'step': int(step), 'mode': 'sequential', 'task': task_name,
                                 'loss': float(loss.item()), 'acc': float(acc)})

            _maybe_print(step, {
                'mode': 'sequential',
                'task': task_name,
                'loss': f"{loss_hist[-1]:.4f}",
                'acc': f"{acc_hist[-1]:.3f}",
                'time_s': f"{time.time()-start:.1f}",
            }, cfg.log_every)

            _maybe_save_intermediate(cfg, model, run_dir, step, ckpts)

    eval_results = {}
    with torch.no_grad():
        model.eval()
        for tn, ds in datasets.items():
            acc = evaluate_task(model, ds.env, device=device, ntrial=max(50, cfg.steps//20),
                                rule_idx=ds.rule_idx, n_rules=ds.n_rules)
            eval_results[tn] = float(acc)

    artifacts = {}
    if csv_rows:
        csv_path = run_dir / "run_curriculum.csv"
        write_csv_rows(csv_path, csv_rows)
        artifacts["curriculum_rows"] = csv_path

    return TrainResult(loss_hist, acc_hist, ema_hist, steps_hist, eval_results, artifacts=artifacts)

def train_bandit(cfg, datasets: Dict[str, Any], bandit, model: torch.nn.Module, opt, device, run_dir: Path) -> TrainResult:
    env0 = next(iter(datasets.values())).env
    n_actions = env0.action_space.n

    loss_hist: List[float] = []
    acc_hist: List[float] = []
    steps_hist: List[int] = []
    ema_hist = {t: [] for t in cfg.tasks}
    base_ema_per_task = {t: None for t in cfg.tasks}

    start = time.time()
    ckpts = deque()

    global_update = 0
    csv_rows: List[Dict[str, Any]] = []

    def do_pg(task_name: str, ds, update_model: bool):
        nonlocal global_update
        sigma_x = getattr(cfg, 'sigma_x', 0.0)
        alpha = getattr(cfg, 'dt', 20.0) / 100.0
        x, y_labels, T, B = sample_batch_with_decision(ds, device=device, sigma_x=sigma_x, alpha=alpha)
        opt.zero_grad()

        # ----- pre-update forward pass -----
        outs_pre = model(x)
        logits_pre = outs_pre["ring_logits"]
        h_pre = outs_pre["h"]

        # Training loss: CE on all non-padding frames
        loss_pre = ce_yang_cmask(logits_pre, y_labels, decision_weight=5.0)  # Yang's c_mask weighting

        # Optional hidden activity regularization
        if getattr(model.hp, "l1_h", 0.0) > 0 or getattr(model.hp, "l2_h", 0.0) > 0:
            loss_pre = loss_pre + activity_reg(
                h_pre,
                getattr(model.hp, "l1_h", 0.0),
                getattr(model.hp, "l2_h", 0.0),
            )

        # Decision-only metric (for accuracy + logging)
        _, logits2d, _ = ce_loss_on_decision_frames(logits_pre, y_labels, n_actions)

        # ----- update RNN if requested -----
        if update_model:
            # Optional weight L1 penalty
            if getattr(cfg, 'l1_weight', 0.0) > 0:
                loss_pre = loss_pre + l1_weight_penalty(
                    model, cfg.l1_weight, getattr(cfg, 'l1_on', 'no-bias')
                )

            # Optional distance penalty on recurrent weights
            if getattr(cfg, 'distance_penalty', False) and getattr(cfg, 'distance_weight', 0.0) > 0:
                loss_pre = loss_pre + distance_penalty(
                    model,
                    cfg.distance_weight,
                    getattr(cfg, 'distance_power', 1.0),
                )

            loss_pre.backward()
            if getattr(cfg, 'grad_clip_mode', 'norm') == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            # Optional proximal L1 step
            if getattr(cfg, 'prox_l1_weight', 0.0) > 0:
                apply_prox_l1_(
                    model,
                    cfg.prox_l1_weight,
                    cfg.lr,
                    getattr(cfg, 'prox_l1_on', 'no-bias'),
                )

            # Reapply any brain connectivity constraints
            if hasattr(model, 'reapply_brain_constraints_'):
                model.reapply_brain_constraints_()

        # ----- post-update loss + reward for bandit -----
        with torch.no_grad():
            outs_post = model(x)
            logits_post = outs_post["ring_logits"]
            loss_post = ce_yang_cmask(logits_post, y_labels, decision_weight=5.0)  # Yang's c_mask weighting

            tau = float(x.shape[0])
            base = loss_pre.item() - loss_post.item()

            # Programmatic normalization of the improvement
            if cfg.pg_normalize == 'time':
                rhat = base / tau
            elif cfg.pg_normalize == 'ema':
                ema = base_ema_per_task[task_name]
                denom = (abs(ema) if ema is not None else abs(base)) + 1e-8
                rhat = base / denom
                alpha = getattr(cfg, 'pg_ema_alpha', 0.9)
                base_ema_per_task[task_name] = (
                    alpha * (ema if ema is not None else abs(base))
                    + (1 - alpha) * abs(base)
                )
            else:
                rhat = base

            # Accuracy for logging
            trial_acc = batch_decision_accuracy_seqwise(
                logits2d, y_labels.reshape(-1), T, B
            )

            global_update += 1
            steps_hist.append(global_update)
            loss_hist.append(loss_pre.item())
            acc_hist.append(float(trial_acc))

            # per-task EMA of accuracy (unchanged)
            for tn in cfg.tasks:
                ema_hist[tn].append(
                    float(trial_acc)
                    if tn == task_name
                    else (ema_hist[tn][-1] if ema_hist[tn] else 0.0)
                )

            return rhat  # raw (normalized) reward signal for this task

    # ----- Main bandit loop -----
    step = 0
    while step < cfg.steps:
        # Bandit chooses a task (arm) via its own sampling method
        arm_idx, task_name, p_arm = bandit.sample()

        # Decide how many steps to run this task
        if cfg.bandit_schedule == 'block':
            n_inner = cfg.steps_per_block
        else:
            n_inner = 1

        # Train for n_inner steps
        running_rhat = 0.0
        count_rhat = 0

        for _ in range(n_inner):
            step += 1
            if step > cfg.steps:
                break

            rhat = do_pg(task_name, datasets[task_name], update_model=True)
            if rhat is not None:
                running_rhat += rhat
                count_rhat += 1

            # Logging for curriculum CSV / printing
            csv_rows.append({
                'step': int(step),
                'mode': 'bandit',
                'task': task_name,
                'rhat': float(rhat) if rhat is not None else None,
                'bandit_reward': None, # filled later if block
                'loss': float(loss_hist[-1]) if loss_hist else None,
                'acc': float(acc_hist[-1]) if acc_hist else None,
            })

            _maybe_print(step, {
                'mode': 'bandit',
                'task': task_name,
                'loss': f"{loss_hist[-1]:.4f}" if loss_hist else "nan",
                'acc': f"{acc_hist[-1]:.3f}" if acc_hist else "nan",
                'time_s': f"{time.time()-start:.1f}",
            }, cfg.log_every)

            _maybe_save_intermediate(cfg, model, run_dir, step, ckpts)

        # Update bandit at the end of the block/step step
        if count_rhat > 0:
            avg_rhat = running_rhat / count_rhat
            bandit_reward = bandit.scale_reward(avg_rhat)
            bandit.update(arm_idx, p_arm, bandit_reward)
            
            # Retroactively log reward for the block rows? 
            # Or just log it on the last step row? Let's just update the last row.
            if len(csv_rows) > 0:
                csv_rows[-1]['bandit_reward'] = float(bandit_reward)

    # Final eval
    eval_results = {}
    with torch.no_grad():
        model.eval()
        for tn, ds in datasets.items():
            acc = evaluate_task(model, ds.env, device=device, ntrial=max(50, cfg.steps//20),
                                rule_idx=ds.rule_idx, n_rules=ds.n_rules)
            eval_results[tn] = float(acc)

    artifacts = {}
    if csv_rows:
        csv_path = run_dir / "run_curriculum.csv"
        write_csv_rows(csv_path, csv_rows)
        artifacts["curriculum_rows"] = csv_path

    return TrainResult(
        loss_hist, acc_hist, ema_hist, steps_hist,
        eval_results,
        artifacts=artifacts,
    )








# train foraging - inspired from developmental cogneuro
def train_foraging(
    cfg,
    datasets: Dict[str, Any],
    forager: MVTCurriculumController,
    model: torch.nn.Module,
    opt,
    device,
    run_dir: Path,
    scaffolder=None,
) -> TrainResult:
    """
    MVT-style foraging curriculum:
    - scalar reward r_t = Δloss(task) = prev_loss(task) - current_loss(task)
    - MVTCurriculumController decides stay/leave + travel
    - task choice after travel is uniform among remaining tasks
    """
    env0 = next(iter(datasets.values())).env
    n_actions = env0.action_space.n

    loss_hist: List[float] = []
    acc_hist: List[float] = []
    steps_hist: List[int] = []
    ema_hist = {t: [] for t in cfg.tasks}
    csv_rows: List[Dict[str, Any]] = []

    # For reward computation per task (Δ decision CE)
    prev_dec_ce_per_task: Dict[str, float] = {}
    # prev_loss_per_task: Dict[str, float] = {}

    # For foraging plots
    rho_hist: List[float] = []
    P_hist: List[float] = []
    action_hist: List[int] = []           # -1 = travel, 0 = stay, 1 = leave
    task_hist: List[Optional[str]] = []   # which task / 'TRAVEL'

    start = time.time()
    global_step = 0          # counts training + travel
    train_step = 0           # counts *only* training updates
    ckpts = deque()


    # ---- helper to get candidates (scaffolding aware) ----
    def _get_candidates(current_train_step: int, last_task: Optional[str] = None) -> List[str]:
        # 1. Base candidates: all tasks or only living ones
        alive = [t for t in list(cfg.tasks) if getattr(datasets[t], "done", False) is False]
        if not alive: alive = list(cfg.tasks)
        
        # 2. Filter by scaffolding
        if scaffolder is not None:
            active = scaffolder.get_active_tasks(current_train_step)
            # Intersection
            candidates = [t for t in alive if t in active]
            # Fallback if intersection empty (shouldn't happen unless bad config)
            if not candidates:
                # Should we default to *all* active, or *all* alive?
                # Default to active (maybe alive ones finished?)
                candidates = active
                if not candidates: candidates = alive # Desperate fallback
        else:
            candidates = alive
            
        # 3. Filter last task (no immediate re-pick same task usually? 
        # Actually standard logic was: [t for t in alive if t != last_task] or alive
        final = [t for t in candidates if t != last_task]
        return final if final else candidates

    # ---- initial task choice ----
    rng = np.random.RandomState(cfg.seed)
    current_task = rng.choice(_get_candidates(0, None))
    forager.start_new_task(current_task)

    # Setup value function
    value_fn = make_value_function(
        getattr(cfg, 'value_func', 'identity'),
        scale=getattr(cfg, 'value_scale', 0.1),
        loss_weight=getattr(cfg, 'value_loss_weight', 2.25)
    )

    # helper to pick a new task uniformly (optionally avoiding current)
    # REPLACED by inline logic using _get_candidates above, but identifying where it is used.
    # The original loop used _pick_new_task. We will redefine it to use our helper.
    def _pick_new_task(last_task: Optional[str]) -> str:
        # Use current train_step
        cands = _get_candidates(train_step, last_task)
        return rng.choice(cands)

    while train_step < cfg.steps:
        global_step += 1  # advance global time (training + travel)

        # --------------- TRAVEL PHASE ---------------
        if forager.in_travel():
            # Optimization: Downsample logging during travel to prevent OOM
            # with high travel steps (e.g. 17M steps -> 17M rows/list items)
            # Log only every 100 steps or at boundaries (start/end)
            is_travel_start = (forager.travel_remaining == forager.travel_steps)
            is_travel_end = (forager.travel_remaining <= 1)
            should_log = (global_step % 100 == 0) or is_travel_start or is_travel_end

            info_t = forager.step_travel()

            if should_log:
                rho_hist.append(float(info_t["rho"]))
                P_hist.append(float(info_t["P"]))
                action_hist.append(int(info_t["action"]))   # -1
                task_hist.append("TRAVEL")

                # do *not* change train_step or steps_hist here:
                # no gradient update happened
            
                loss_hist.append(loss_hist[-1] if loss_hist else 0.0)
                acc_hist.append(acc_hist[-1] if acc_hist else 0.0)
                for tn in cfg.tasks:
                    ema_hist[tn].append(ema_hist[tn][-1] if ema_hist[tn] else 0.0)

                csv_rows.append({
                    "step": int(global_step),        # global time
                    "train_step": int(train_step),   # how many updates done so far
                    "regime": "foraging",
                    "mode": "travel",
                    "task": info_t["current_task"],
                    "loss": None,
                    "acc": None,
                    "reward": float(info_t["reward"]),
                    "rho": float(info_t["rho"]),
                    "P": float(info_t["P"]),
                    "block_steps": int(info_t["block_steps"]),
                    "decision": int(info_t["action"]),
                })

            # When travel ends, sample a new task and reset local state
            if forager.travel_remaining == 0:
                new_task = _pick_new_task(forager.current_task)
                current_task = new_task
                forager.start_new_task(new_task)

            continue  # next *global* step (still may be travel or train)

        # --------------- TRAINING PHASE ---------------
        train_step += 1                 # count a real gradient step
        steps_hist.append(train_step)   # this matches joint/sequential semantics

        ds = datasets[current_task]
        sigma_x = getattr(cfg, 'sigma_x', 0.0)
        alpha = getattr(cfg, 'dt', 20.0) / 100.0
        x, y_labels, T, B = sample_batch_with_decision(ds, device=device, sigma_x=sigma_x, alpha=alpha)
        opt.zero_grad()


        outs = model(x)
        logits = outs["ring_logits"]
        h_seq = outs["h"]

        # 1) Pure *task* loss over all non-padding frames (used for training)
        loss = ce_yang_cmask(logits, y_labels, decision_weight=5.0)  # Yang's c_mask weighting

        # 2) Decision-frame CE (used for reward + dec accuracy)
        dec_ce_loss, logits2d, _ = ce_loss_on_decision_frames(
            logits, y_labels, n_actions
        )

        # Decision-only metric
        _, logits2d, _ = ce_loss_on_decision_frames(logits, y_labels, n_actions)

        # Activity regularization
        if getattr(model.hp, "l1_h", 0.0) > 0 or getattr(model.hp, "l2_h", 0.0) > 0:
            loss = loss + activity_reg(
                h_seq,
                getattr(model.hp, "l1_h", 0.0),
                getattr(model.hp, "l2_h", 0.0),
            )

        # L1 on weights
        if getattr(cfg, "l1_weight", 0.0) > 0:
            loss = loss + l1_weight_penalty(
                model, cfg.l1_weight, getattr(cfg, "l1_on", "no-bias")
            )

        # Distance penalty
        if getattr(cfg, "distance_penalty", False) and getattr(cfg, "distance_weight", 0.0) > 0:
            loss = loss + distance_penalty(
                model,
                cfg.distance_weight,
                getattr(cfg, "distance_power", 1.0),
            )

        loss.backward()
        if getattr(cfg, 'grad_clip_mode', 'norm') == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if getattr(cfg, "prox_l1_weight", 0.0) > 0:
            apply_prox_l1_(
                model,
                cfg.prox_l1_weight,
                cfg.lr,
                getattr(cfg, "prox_l1_on", "no-bias"),
            )

        if hasattr(model, "reapply_brain_constraints_"):
            model.reapply_brain_constraints_()

        with torch.no_grad():
            acc = batch_decision_accuracy_seqwise(
                logits2d, y_labels.reshape(-1), T, B
            )

        # Histories
        loss_val = float(loss.item())
        loss_hist.append(loss_val)
        acc_hist.append(float(acc))

        # Per-task EMA of accuracy (like sequential)
        # Update current task
        ema_prev = ema_hist[current_task][-1] if ema_hist[current_task] else None
        alpha_ema = getattr(cfg, "pg_ema_alpha", 0.9)
        new_ema = (alpha_ema * ema_prev + (1 - alpha_ema) * float(acc)) if ema_prev is not None else float(acc)

        for tn in cfg.tasks:
            ema_hist[tn].append(
                new_ema
                if tn == current_task
                else (ema_hist[tn][-1] if ema_hist[tn] else 0.0)
            )

        # ----- reward = learning progress in *decision CE* for this task -----
        if dec_ce_loss is not None:
            dec_val = float(dec_ce_loss.item())
            prev_dec = prev_dec_ce_per_task.get(current_task)
            if prev_dec is None:
                # First time we see this task: no previous CE → zero reward
                reward = 0.0
            else:
                reward = prev_dec - dec_val  # positive = improvement
            prev_dec_ce_per_task[current_task] = dec_val
        else:
            # No valid decision frames in this batch; treat as neutral for forager
            reward = 0.0

        # ----- Forager step -----
        # Transform objective reward (learning progress) into subjective utility
        subj_reward = value_fn(reward)
        info_t = forager.step_training(subj_reward)
        rho_hist.append(float(info_t["rho"]))
        P_hist.append(float(info_t["P"]))
        action_hist.append(int(info_t["action"]))   # 0 stay, 1 leave
        task_hist.append(current_task)

        csv_rows.append({
            "step": int(global_step),       # global time axis
            "train_step": int(train_step),  # number of updates
            "regime": "foraging",
            "mode": "train",
            "task": current_task,
            "loss": loss_val,
            "loss": loss_val,
            "acc": float(acc),
            "ema": float(new_ema),
            "reward": float(reward),        # log raw reward (objective)
            "subj_reward": float(subj_reward), # log subjective utility (used for decision)
            "rho": float(info_t["rho"]),
            "P": float(info_t["P"]),
            "block_steps": int(info_t["block_steps"]),
            "decision": int(info_t["action"]),
            "note": info_t.get("note", ""),
        })

        _maybe_print(
            train_step,
            {
                "mode": "foraging",
                "task": current_task,
                "loss": f"{loss_val:.4f}",
                "acc": f"{float(acc):.3f}",
                "rho": f"{info_t['rho']:.4f}",
                "P": f"{info_t['P']:.4f}",
                "time_s": f"{time.time() - start:.1f}",
                "gstep": str(global_step),   # optional: show global step too
            },
            cfg.log_every,
        )

        _maybe_save_intermediate(cfg, model, run_dir, train_step, ckpts)
        # If we chose to leave, travel will start on next iteration
        # (forager.travel_remaining already set inside controller)

    # ----- Final eval -----
    eval_results = {}
    with torch.no_grad():
        model.eval()
        for tn, ds in datasets.items():
            acc = evaluate_task(model, ds.env, device=device, ntrial=max(50, cfg.steps // 20),
                                rule_idx=ds.rule_idx, n_rules=ds.n_rules)
            eval_results[tn] = float(acc)

    artifacts = {}
    if csv_rows:
        csv_path = run_dir / "run_curriculum.csv"
        write_csv_rows(csv_path, csv_rows)
        artifacts["curriculum_rows"] = csv_path

    return TrainResult(
        loss_hist,
        acc_hist,
        ema_hist,
        steps_hist,
        eval_results,
        artifacts=artifacts,
    )

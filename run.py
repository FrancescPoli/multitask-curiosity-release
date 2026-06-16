#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Dict
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from paths import get_logs_dir


from curiosity.data import make_dataset
from curiosity.logging_io import ensure_run_dir
from curiosity.train.loops import train_joint, train_sequential, train_bandit, train_foraging
from curiosity.policies import Exp3S
from curiosity.checkpoints import save_checkpoint, save_state_dict, write_json

# Khona-style model
import curiosity.models.yang19_khona as tyk  # expects Yang19KhonaModel, HParams, make_optimizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class Config:
    logdir: str = str(get_logs_dir())
    run_name: str = 'run'
    tasks: List[str] = None
    dt: float = 20.0
    activation: str = 'relu'      # Yang19 default
    hidden: int = 256             # Yang19 default
    batch_size: int = 32          # User preference (Yang 64, but 32 is acceptable)
    steps: int = 2000
    steps_per_block: int = 100
    lr: float = 2e-3
    grad_clip: float = 1.0
    grad_clip_mode: Literal['norm', 'value'] = 'value'
    # Weight sparsity controls
    l1_weight: float = 0.0
    prox_l1_weight: float = 0.0
    l1_on: str = 'no-bias'
    prox_l1_on: str = 'no-bias'
    l1_h: float = 0.0
    l2_h: float = 0.0
    distance_penalty: bool = False
    distance_weight: float = 0.0
    distance_power: float = 1.0
    seed: int = 0
    bandit: Literal['none','pg','spg'] = 'none'
    bandit_schedule: Literal['step','block'] = 'block'
    bandit_eta: float = 5e-3
    bandit_eps: float = 0.05
    bandit_share: float = 0.0
    bandit_reservoir: int = 5000
    bandit_qlo: float = 0.2
    bandit_qhi: float = 0.8
    pg_normalize: Literal['time','none'] = 'time'
    reward_scale: Literal['quantile','tanh','zscore','none'] = 'quantile'
    log_policy: bool = False
    log_every: int = 100
    use_separate_input: bool = False  # default to Yang-style fused input
    # Initialization
    w_rec_init: str = 'diag'          # diag, randortho, randgauss
    w_rec_noise: float = 0.0          # noise added to w_rec initialization
    sigma_x: float = 0.01             # input noise (Yang default 0.01)
    foraging: str = 'none'            # 'none' or 'mvt'
    forage_alpha_local: float = 0.03
    forage_beta_global: float = 0.003
    forage_min_block_steps: int = 10
    forage_travel_steps: int = 50
    forage_travel_steps: int = 50
    forage_eps: float = 0.0
    forage_temperature: float = 0.0
    dataset_mode: str = 'episode'
    rule_encoding: str = 'onehot' # 'onehot' or 'lowrank'
    rule_dim_low: int = 4
    rule_dim_out: Optional[int] = None # NEW: Override output dimension (e.g. 100 for future tasks)

    # intermediate saving
    save_every: int = 0          # 0 = only final snapshot
    max_checkpoints: int = 0     # 0 = keep all; >0 = rolling window
    # Value function
    value_func: str = 'tanh_asym'
    value_scale: float = 0.1
    value_loss_weight: float = 2.25
    # Scaffolding
    scaffolding: bool = False
    scaffold_factor: float = 1.0
    scaffold_mode: str = 'cumulative'  # 'cumulative' or 'disjoint'
    scaffold_num_groups: int = 5

def build_model_opt_hp(
    datasets: Dict[str, any],
    hidden: int,
    activation: str,
    lr: float,
    seed: int,
    dt: float,
    *,
    l1_h: float = 0.0,          
    l2_h: float = 0.0,
    w_rec_init: str = 'diag',
    w_rec_noise: float = 0.0,  # Yang's default (no init noise)
    mask_path=None,
    mask_as_init=False,
    use_dales=False,
    exc_frac=0.8,
    target_sr=None,  # Match Yang: no spectral radius rescaling
    dist_path: Optional[str] = None,
    use_separate_input: bool = True,
    rule_dim: Optional[int] = None, # NEW: Explicit rule dimension
):
    env0 = next(iter(datasets.values())).env
    obs_dim = env0.observation_space.shape[0]
    n_actions = env0.action_space.n

    # verify all envs have same obs_dim
    for ds in datasets.values():
        if ds.env.observation_space.shape[0] != obs_dim:
            raise ValueError("All tasks must share the same observation dimension for separate-input layout.")

    # ---- Infer rule layout from obs_dim and number of tasks ----
    n_tasks = len(datasets)  # we assume one rule unit per task

    if use_separate_input:
        # Use explicit rule_dim if provided, else fall back to n_tasks
        n_rule = rule_dim if rule_dim is not None else n_tasks
        rule_start = obs_dim - n_rule
        if rule_start <= 0:
            raise ValueError(
                f"Cannot infer [sensory]/[rule] split: obs_dim={obs_dim}, n_rule={n_rule} "
                "→ rule_start <= 0. Make sure the last n_rule dims of the observation are the rule vector."
            )
    else:
        n_rule = 0
        rule_start = 0

    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hp = tyk.HParams(
        n_input=obs_dim,
        n_rnn=hidden,
        n_ring=n_actions,            # CE over ring classes
        dt=dt,
        activation=activation,
        learning_rate=lr,
        seed=seed,
        l1_h=l1_h,
        l2_h=l2_h,
        use_separate_input=use_separate_input,
        rule_start=rule_start,
        n_rule=n_rule,
        w_rec_init=w_rec_init,
        w_rec_noise=w_rec_noise,
    )
    model = tyk.Yang19KhonaModel(hp, mask_path=mask_path, mask_as_init=mask_as_init,
                        use_dales=use_dales, exc_frac=exc_frac, target_sr=target_sr,
                        dist_path=dist_path).to(DEVICE).train()
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError('No trainable parameters found in model. Check Yang19KhonaModel initialization and brain constraints setup.')
    opt = tyk.make_optimizer(params, hp)
    return model, opt, hp

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', type=str, default=str(get_logs_dir()))
    ap.add_argument('--run-name', type=str, default='run')
    ap.add_argument('--tasks', nargs='+', default=['yang19.go-v0', 'yang19.rtgo-v0', 'yang19.dm1-v0', 'yang19.ctxdm1-v0'])
    ap.add_argument('--dt', type=float, default=20.0) #
    ap.add_argument('--activation', type=str, default='relu', choices=['softplus','tanh','relu'])
    ap.add_argument('--hidden', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=64)  # Yang's default
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--steps-per-block', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--grad-clip-mode', type=str, default='value', choices=['norm', 'value'])
    # Initialization
    ap.add_argument('--w-rec-init', type=str, default='randortho', choices=['diag', 'randortho', 'randgauss'])  # Yang's default
    ap.add_argument('--w-rec-noise', type=float, default=0.0)  # Yang uses 0.0 (only sigma_rec for noise)
    ap.add_argument('--sigma-x', type=float, default=0.01)     # Input noise (Yang default 0.01)
    ap.add_argument('--l1-weight', type=float, default=0.0, help='L1 penalty coefficient on weights (added to loss)')
    ap.add_argument('--prox-l1-weight', type=float, default=0.0, help='Proximal L1 coefficient on weights (soft-threshold after each step)')
    ap.add_argument('--l1-on', type=str, default='no-bias', choices=['all','no-bias','recurrent-only'], help='Which parameters to apply L1 to')
    ap.add_argument('--prox-l1-on', type=str, default='recurrent-only', choices=['all','no-bias','recurrent-only'], help='Which parameters to apply proximal L1 to')
    ap.add_argument('--l1-h', type=float, default=0.0)                   # activity ℓ1
    ap.add_argument('--l2-h', type=float, default=0.0)                   # activity ℓ2
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mask-path', type=str, default=None)
    ap.add_argument('--mask-as-init', action='store_true')
    ap.add_argument('--use-dales', action='store_true')
    ap.add_argument('--exc-frac', type=float, default=0.8)
    ap.add_argument('--target-sr', type=float, default=0.5)
    #Make sure 𝐷 is on a sensible scale (e.g., z-scored or max-normalized to 1); otherwise  𝜆 becomes hard to tune.
    ap.add_argument('--distance-penalty', action='store_true', help='Enable |Wrec|-weighted distance penalty using a provided HxH matrix D')
    ap.add_argument('--distance-path', type=str, default=None, help='Path to HxH .npy/.pt distance matrix aligned to hidden size')
    ap.add_argument('--distance-weight', type=float, default=0.0, help='Lambda for distance penalty (coeff * sum(|Wrec| * D^p))')
    ap.add_argument('--distance-power', type=float, default=1.0, help='Exponent p on distances: use D^p')
    ap.add_argument('--sequential-blocks', '--sequential', action='store_true')
    ap.add_argument('--bandit', type=str, default='none', choices=['none','pg','spg'])
    ap.add_argument('--bandit-schedule', type=str, choices=['step','block'], default='block')
    ap.add_argument('--bandit-eta', type=float, default=5e-3)
    ap.add_argument('--bandit-eps', type=float, default=0.05)
    ap.add_argument('--bandit-share', type=float, default=0.0)
    ap.add_argument('--bandit-reservoir', type=int, default=5000)
    ap.add_argument('--bandit-qlo', type=float, default=0.2)
    ap.add_argument('--bandit-qhi', type=float, default=0.8)
    ap.add_argument('--pg-normalize', type=str, choices=['time','ema','none'], default='time')
    ap.add_argument('--reward-scale', type=str, choices=['quantile','tanh','zscore','none'], default='quantile')
    ap.add_argument('--log-policy', action='store_true')
    ap.add_argument('--log-every', dest='log_every', type=int, default=100)
        # Foraging curriculum (MVT-style stay/leave)
    ap.add_argument('--foraging', type=str, default='none',
                    choices=['none', 'mvt'],
                    help="Use MVT-style foraging curriculum instead of bandit/joint/sequential.")
    ap.add_argument('--forage-alpha-local', type=float, default=0.03)
    ap.add_argument('--forage-beta-global', type=float, default=0.003)
    ap.add_argument('--forage-min-block-steps', type=int, default=10)
    ap.add_argument('--forage-travel-steps', type=int, default=50)
    ap.add_argument('--forage-eps', type=float, default=0.0)
    ap.add_argument('--forage-temperature', type=float, default=0.0, help="Temperature for stochastic MVT decision (0.0 = deterministic).")
    ap.add_argument('--dataset-mode', type=str, default='episode', choices=['episode', 'stream'],
                    help="Training regime: 'episode' (independent trials, cold start) or 'stream' (continuous, warm start).")
    ap.add_argument('--use-separate-input', action='store_true', default=False,
                    help='Use separate sensory + rule projections (Khona-style). Default: False (Yang-style fused input).')
    ap.add_argument('--rule-encoding', type=str, default='onehot', choices=['onehot', 'lowrank'],
                    help="Rule encoding scheme: 'onehot' (canonical) or 'lowrank' (random projected dense vectors).")
    ap.add_argument('--rule-dim-low', type=int, default=4, help="Dimensionality of the low-rank bottleneck for random rule projections.")
    ap.add_argument('--rule-dim-out', type=int, default=None, help="Output dimension for rule vectors (overrides n_tasks)")

    ap.add_argument('--save-every', type=int, default=0,
        help='If >0, save intermediate state_dict every N (outer) training steps.')
    ap.add_argument('--max-checkpoints', type=int, default=100,
        help='If >0, keep at most this many intermediate checkpoints (rolling). Default 100.')
    ap.add_argument('--value-func', type=str, default='tanh_asym', choices=['identity', 'tanh_asym'])
    ap.add_argument('--value-scale', type=float, default=0.1)
    ap.add_argument('--value-loss-weight', type=float, default=2.25)
    # Scaffolding
    ap.add_argument('--scaffolding', action='store_true', help="Enable complexity-weighted curriculum diet.")
    ap.add_argument('--scaffold-factor', type=float, default=1.0, help="Multiplier for duration of subsequent groups.")
    ap.add_argument('--scaffold-mode', type=str, default='cumulative', choices=['cumulative', 'disjoint'], help="'cumulative': tasks accumulate. 'disjoint': only current group active.")
    ap.add_argument('--scaffold-num-groups', type=int, default=5, help="Number of balanced complexity groups.")
    args = ap.parse_args()

    # auto-pick default mask if not provided
    #default_mask = Path(__file__).parent / 'brain' / 'assets' / 'brain_mask.npy'
    #if getattr(args, 'mask_path', None) is None and default_mask.exists():
    #    args.mask_path = str(default_mask)

    # infer hidden if needed
    if args.mask_path is not None:
        A = np.load(args.mask_path)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Mask must be square (H×H); got {A.shape}.")
        H = int(A.shape[0])

        if args.hidden is None:
            # user didn’t set --hidden → adopt mask size
            args.hidden = H
        elif args.hidden != H:
            # user explicitly set a different size → fail fast
            raise ValueError(f"--hidden={args.hidden} does not match mask size {H}. "
                            f"Either omit --hidden or set it to {H}.")
    # no mask → fall back to old default
    if args.hidden is None:
        args.hidden = 100

    cfg = Config(
        logdir=args.logdir,
        run_name=args.run_name,
        tasks=args.tasks,
        dt=args.dt,
        activation=args.activation,
        hidden=args.hidden,
        w_rec_init=args.w_rec_init,
        w_rec_noise=args.w_rec_noise,
        sigma_x=args.sigma_x,
        batch_size=args.batch_size,
        steps=args.steps,
        steps_per_block=args.steps_per_block,
        lr=args.lr,
        grad_clip=args.grad_clip,
        grad_clip_mode=args.grad_clip_mode,
        seed=args.seed,
        bandit=args.bandit,
        bandit_schedule=args.bandit_schedule,
        bandit_eta=args.bandit_eta,
        bandit_eps=args.bandit_eps,
        bandit_share=args.bandit_share,
        bandit_reservoir=args.bandit_reservoir,
        bandit_qlo=args.bandit_qlo,
        bandit_qhi=args.bandit_qhi,
        pg_normalize=args.pg_normalize,
        reward_scale=args.reward_scale,
        log_policy=args.log_policy,
        log_every=args.log_every,
        l1_weight=args.l1_weight,
        prox_l1_weight=args.prox_l1_weight,
        l1_on=args.l1_on,
        prox_l1_on=args.prox_l1_on,
        l1_h=args.l1_h,
        l2_h=args.l2_h,
        distance_penalty=args.distance_penalty,
        distance_weight=args.distance_weight,
        distance_power=args.distance_power,
        use_separate_input=args.use_separate_input,
        foraging=args.foraging,
        forage_alpha_local=args.forage_alpha_local,
        forage_beta_global=args.forage_beta_global,
        forage_min_block_steps=args.forage_min_block_steps,
        forage_travel_steps=args.forage_travel_steps,
        forage_eps=args.forage_eps,
        forage_temperature=args.forage_temperature,
        dataset_mode=args.dataset_mode,
        rule_encoding=args.rule_encoding,
        rule_dim_low=args.rule_dim_low,
        rule_dim_out=args.rule_dim_out,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        value_func=args.value_func,
        value_scale=args.value_scale,
        value_loss_weight=args.value_loss_weight,
        scaffolding=args.scaffolding,
        scaffold_factor=args.scaffold_factor,
        scaffold_mode=args.scaffold_mode,
        scaffold_num_groups=args.scaffold_num_groups,
    )

    if args.distance_penalty and not args.distance_path:
        raise ValueError("--distance-penalty is set but --distance-path is missing")

    run_dir = ensure_run_dir(cfg)
    
    # NEW: Determine if we should append rule inputs.
    # If not using separate inputs (Yang style), we likely want to append one-hot rules
    # if there are multiple tasks, so the model can distinguish them.
    # However, if user_separate_input is True, the model handles it via slicing. 
    # BUT, 'make_dataset' logic I added appends it to the *observation*.
    # If use_separate_input is True, we need the rule part in the observation anyway so the model can slice it.
    # So we ALWAYS append rules if there is more than 1 task, OR if we rely on rule inputs being present.
    # For full safety: Always append one-hot rule vector if we have a list of tasks.
    
    # Exception: if the task itself already provides it? 
    # Neurogym Yang19 tasks do NOT provided it by default (obs dim 33).
    # So we should inject it.
    
    n_tasks = len(cfg.tasks)
    datasets = {}
    
    # Init Rule Encoder (Seed ensures determinism)
    # Init Rule Encoder (Seed ensures determinism)
    from curiosity.data import RuleEncoder
    # If rule_dim_out is set (e.g. 100), we treat it as total capacity for both
    # 1. Total tasks to generate (to lock the seed)
    # 2. Output dimension (to reserve space)
    # This future-proofs against adding more tasks later.
    capacity = cfg.rule_dim_out if cfg.rule_dim_out is not None else n_tasks
    
    rule_enc = RuleEncoder(
        seed=cfg.seed,
        n_tasks=capacity,  # Generate 'capacity' vectors now to lock random stream
        dim_out=capacity,  # Project to 'capacity' dimensions
        encoding=cfg.rule_encoding,
        dim_low=cfg.rule_dim_low
    )
    print(f"[Run] Rule Encoder initialized with capacity {capacity} (dim_out={capacity})")
    
    for i, name in enumerate(cfg.tasks):
        # We pass rule_vector directly
        rule_vec = rule_enc.get_vector(i)
        
        datasets[name] = make_dataset(name, dt=cfg.dt, batch_size=cfg.batch_size, 
                                      rule_vector=rule_vec, # Replaces rule_idx/n_rules logic
                                      dataset_mode=cfg.dataset_mode)

    # --- Normalize steps to mean "Total Gradient Updates" ---
    # User Request: "20000 steps" should mean 20k total updates.
    # Training mode detection
    # NOTE: After fixing catastrophic forgetting, we now use Yang's approach:
    # 1 step = 1 gradient update on 1 randomly sampled task
    # So cfg.steps is the ACTUAL number of updates (no division needed)
    n_tasks = len(cfg.tasks)
    print(f"[Run Config] Training with {n_tasks} tasks using random sampling (Yang-style)")
    print(f"             Total gradient updates: {cfg.steps}")
    print(f"             Each task gets ~{cfg.steps / n_tasks:.0f} updates on average")



    model, opt, hp = build_model_opt_hp(
        datasets,
        cfg.hidden,
        cfg.activation,
        cfg.lr,
        cfg.seed,
        cfg.dt,
        l1_h=cfg.l1_h,
        l2_h=cfg.l2_h,
        w_rec_init=cfg.w_rec_init,
        w_rec_noise=cfg.w_rec_noise,
        mask_path=args.mask_path,
        mask_as_init=args.mask_as_init,
        use_dales=args.use_dales,
        exc_frac=args.exc_frac,
        target_sr=args.target_sr,
        dist_path=args.distance_path,
        use_separate_input=cfg.use_separate_input,
        rule_dim=capacity,  # Correctly pass the full rule dimension (e.g. 100)
    )


    # ---- Train
    scaffolder = None
    if cfg.scaffolding:
        from curiosity.curriculums.scaffolding import ScaffoldCurriculum
        scaffolder = ScaffoldCurriculum(
            tasks=list(cfg.tasks),
            total_steps=cfg.steps,
            factor=cfg.scaffold_factor,
            mode=cfg.scaffold_mode,
            num_groups=cfg.scaffold_num_groups,
            seed=cfg.seed,
        )

    # ---- Save initial state (Step 0)
    ckpt_path_0 = run_dir / 'state_step000000.pt'
    # Minimal meta for init
    meta_init = {
        "cfg": vars(cfg),
        "step": 0
    }
    save_checkpoint(model, ckpt_path_0, meta=meta_init)
    print(f"[init] Saved initial checkpoint: {ckpt_path_0}")

    if cfg.foraging != 'none':
        # MVT-style foraging curriculum
        from curiosity.train.foraging import MVTCurriculumController
        forager = MVTCurriculumController(
            alpha_local=cfg.forage_alpha_local,
            beta_global=cfg.forage_beta_global,
            min_block_steps=cfg.forage_min_block_steps,
            travel_steps=cfg.forage_travel_steps,
            eps=cfg.forage_eps,
            temperature=cfg.forage_temperature,
        )
        result = train_foraging(cfg, datasets, forager, model, opt, DEVICE, run_dir, scaffolder=scaffolder)

    elif cfg.bandit != 'none':
        bandit = Exp3S(
            task_names=cfg.tasks,
            eta=cfg.bandit_eta,
            eps=cfg.bandit_eps,
            share=cfg.bandit_share,
            reservoir=cfg.bandit_reservoir,
            qlo=cfg.bandit_qlo,
            qhi=cfg.bandit_qhi,
            seed=cfg.seed,
            scale_mode=cfg.reward_scale,
        )
        result = train_bandit(cfg, datasets, bandit, model, opt, DEVICE, run_dir, scaffolder=scaffolder)

    else:
        if args.sequential_blocks:
            result = train_sequential(cfg, datasets, model, opt, DEVICE, run_dir, scaffolder=scaffolder)
        else:
            result = train_joint(cfg, datasets, model, opt, DEVICE, run_dir, scaffolder=scaffolder)
            
    # ---- Save weights + metadata
    ckpt_path = run_dir / 'model_last.pt'
    raw_path  = run_dir / 'state_dict.pt'
    meta = {
    "cfg": vars(cfg),
    "hparams": getattr(hp, "__dict__", {}),
    "model_class": "Yang19KhonaModel",
    "model_kwargs": {  # NEW
        "mask_path": args.mask_path,
        "mask_as_init": args.mask_as_init,
        "use_dales": args.use_dales,
        "exc_frac": args.exc_frac,
        "target_sr": args.target_sr,
        "dist_path": args.distance_path if args.distance_penalty else None,  
    },
    }
    save_checkpoint(model, ckpt_path, meta=meta)
    save_state_dict(model, raw_path)
    write_json(meta, run_dir / 'model_meta.json')
    print(f"[saved] {ckpt_path} and {raw_path} (metadata in model_meta.json)")

    print("Done. Artifacts in:", run_dir)

if __name__ == '__main__':
    main()

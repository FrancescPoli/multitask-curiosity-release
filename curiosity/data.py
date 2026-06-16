# curiosity/data.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
try:
    import neurogym as ngym
except Exception as e:
    raise RuntimeError("Install neurogym: pip install neurogym") from e

# NEW: Mod-Cog adapter
try:
    from .modcog_adapter import make_modcog_dataset
except Exception:
    # allow running this file standalone (e.g., from a notebook)
    from curiosity.modcog_adapter import make_modcog_dataset  # type: ignore


@dataclass
class RuleEncoder:
    """
    Deterministic generator for task rule vectors.
    Supports 'onehot' (canonical) and 'lowrank' (random projection) encodings.
    """
    seed: int
    n_tasks: int
    dim_out: int
    encoding: str = 'onehot'
    dim_low: int = 4
    
    def __post_init__(self):
        # We use a local RandomState to ensure determinism regardless of global seed state
        rng = np.random.RandomState(self.seed)
        
        if self.encoding == 'onehot':
            # Standard Identity Matrix
            # If dim_out > n_tasks, we pad with zeros? usually dim_out == n_tasks
            self.matrix = np.eye(self.dim_out, dtype=np.float32)
            # If we requested fewer tasks than dim_out, we just take the first N rows
            # If we requested more tasks than dim_out, standard one-hot fails or wraps.
            # Assuming dim_out >= n_tasks for one-hot
            
        elif self.encoding == 'lowrank':
            # 1. Generate latent vectors in low-dim space (crowded)
            # Shape: (n_tasks, dim_low)
            Z = rng.randn(self.n_tasks, self.dim_low).astype(np.float32)
            # Normalize latents to be on hypersphere?
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
            
            # 2. Random Projection Matrix (dim_low -> dim_out)
            W_proj = rng.randn(self.dim_low, self.dim_out).astype(np.float32)
            # Optional: Orthogonalize W_proj? 
            # For random projection, Gaussian is fine.
            
            # 3. Project
            V = Z @ W_proj # (n_tasks, dim_out)
            
            # 4. Normalize output vectors to have unit norm, 
            # matching the scale of one-hot vectors (norm=1)
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            
            self.matrix = V
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def get_vector(self, idx: int) -> np.ndarray:
        if idx >= len(self.matrix):
             raise ValueError(f"Task index {idx} out of bounds for RuleEncoder (size {len(self.matrix)})")
        return self.matrix[idx]


class RuleAugmentedEnv:
    """
    Wraps a NeuroGym environment to automatically append a static rule vector
    to the observations (self.ob).
    This ensures both the formatting in training loops AND evaluation loops (which access env.ob)
    see the correct augmented input.
    """
    def __init__(self, env, rule_vector: np.ndarray):
        self.env = env
        self.rule_vector = rule_vector.astype(np.float32)
        
        # Calculate new shape
        import gym
        from gym import spaces
        orig_shape = env.observation_space.shape
        new_shape = (orig_shape[0] + self.rule_vector.shape[0],)
        
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=new_shape, dtype=np.float32
        )
        self.action_space = env.action_space
        
    def __getattr__(self, name):
        return getattr(self.env, name)
        
    def new_trial(self, **kwargs):
        self.env.new_trial(**kwargs)
        
    @property
    def ob(self):
        raw = self.env.ob
        if raw is None: return None
        # raw shape (T, D_orig)
        T = raw.shape[0]
        # Tile rule vector to (T, D_rule)
        rules = np.tile(self.rule_vector, (T, 1))
        return np.concatenate([raw, rules], axis=-1)
        
    @property
    def gt(self):
        return self.env.gt


@dataclass
class YangEpisodeDataset:
    """
    Generates batches of independent episodes (trials) instead of a continuous stream.
    Each item in the batch is a single trial starting from t=0.
    Trials are padded to the maximum length in the batch.
    """
    task_name: str
    dt: float = 20.0
    batch_size: int = 16
    timing: Optional[dict] = None
    rule_idx: Optional[int] = None
    n_rules: Optional[int] = None
    rule_vector: Optional[np.ndarray] = None # NEW: Pre-computed vector

    def __post_init__(self):
        env_kwargs = {'dt': self.dt}
        if self.timing is not None:
            env_kwargs['timing'] = self.timing
            
        if 'poli' in self.task_name:
            env_kwargs['dim_ring'] = 16
        
        # We use a raw gym/neurogym env, not a dataset wrapper
        import neurogym as ngym
        self.env = ngym.make(self.task_name, **env_kwargs)
        
        # Resolve Rule Vector
        # If we have legacy rule_idx, convert to onehot vector
        final_rule_vec = None
        if self.rule_vector is not None:
             final_rule_vec = self.rule_vector
        elif self.rule_idx is not None and self.n_rules is not None:
             v = np.zeros(self.n_rules, dtype=np.float32)
             v[self.rule_idx] = 1.0
             final_rule_vec = v
             
        # Wrap Environment if needed
        if final_rule_vec is not None:
            self.env = RuleAugmentedEnv(self.env, final_rule_vec)

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        obs_list = []
        gt_list = []
        lengths = []
        
        # Generate N independent trials
        for i in range(self.batch_size):
            self.env.new_trial()
            # Copy to ensure no reference issues (though usually safe in ngym)
            ob = self.env.ob.copy() # (T, D) - ALREADY AUGMENTED by Wrapper
            gt = self.env.gt.copy() # (T,)
            
            obs_list.append(ob)
            gt_list.append(gt)
            lengths.append(ob.shape[0])
            
        max_len = max(lengths)
        dim = obs_list[0].shape[1]
        
        # Pad and Stack
        # Final shape: (T_max, B, Dim)
        x_batch = np.zeros((max_len, self.batch_size, dim), dtype=np.float32)
        y_batch = np.full((max_len, self.batch_size), -100, dtype=np.int64) # -100 padding for ignore (PyTorch default)
        
        for i in range(self.batch_size):
            L = lengths[i]
            x_batch[:L, i, :] = obs_list[i]
            y_batch[:L, i] = gt_list[i]
            
        # No manual appending needed anymore!
            
        return x_batch, y_batch


@dataclass
class TaskDataset:
    task_name: str
    dt: float = 20.0
    batch_size: int = 16
    seq_target_trials: int = 1  # Match Yang: 1 trial per sequence (was 3)
    timing: Optional[dict] = None
    seq_len: Optional[int] = None # Added explicit override
    rule_idx: Optional[int] = None  # NEW: Index of this task for rule input
    n_rules: Optional[int] = None   # NEW: Total number of rule inputs
    rule_vector: Optional[np.ndarray] = None # NEW

    def __post_init__(self):
        # NeuroGym Dataset (Yang'19-style): infer a good sequence length
        if self.seq_len is None:
            max_steps = estimate_trial_len_steps(self.task_name, dt=self.dt, timing=self.timing, n=64)
            self.seq_len = self.seq_target_trials * max_steps + 5
            
        env_kwargs = {'dt': self.dt}
        if self.timing is not None:
            env_kwargs['timing'] = self.timing
            
        # 1. Create Base Dataset (which creates the internal env)
        import neurogym as ngym
        self.ds = ngym.Dataset(self.task_name, env_kwargs=env_kwargs, batch_size=self.batch_size, seq_len=self.seq_len)
        self.env = self.ds.env
        
        # Resolve Rule Vector
        final_rule_vec = None
        if self.rule_vector is not None:
             final_rule_vec = self.rule_vector
        elif self.rule_idx is not None and self.n_rules is not None:
             v = np.zeros(self.n_rules, dtype=np.float32)
             v[self.rule_idx] = 1.0
             final_rule_vec = v
             
        # Wrap Environment if needed
        # Note: ngym.Dataset wraps env internally somewhat, but exposes .env
        # If we replace self.env with our wrapper, we need to ensure self.ds uses it?
        # ngym.Dataset.__call__ uses self.env.
        # So we patch self.ds.env AND self.env
        if final_rule_vec is not None:
            wrapper = RuleAugmentedEnv(self.env, final_rule_vec)
            self.ds.env = wrapper
            self.env = wrapper

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        # self.ds() calls self.env.new_trial() and uses self.env.ob/gt
        # Since we patched self.ds.env, it should work!
        ob, gt = self.ds()
                
        return ob, gt


def make_dataset(task_name: str, dt: float, batch_size: int, seq_len: Optional[int] = None, 
                 rule_idx: Optional[int] = None, n_rules: Optional[int] = None,
                 rule_vector: Optional[np.ndarray] = None,
                 dataset_mode: str = 'episode'):
    """Factory that now supports:
       - 'episode' mode (DEFAULT): YangEpisodeDataset (independent trials, cold start).
       - 'stream' mode: TaskDataset (continuous stream, warm start).
       - Mod-Cog (Khona): Always separate adapter logic.
       
       params:
         rule_idx (int): If provided, appends a one-hot vector to the observation identifying the task.
         n_rules (int): Total size of the rule vector.
         rule_vector (np.ndarray): If provided, explicitly appends this vector (overrides rule_idx).
         dataset_mode (str): 'episode' or 'stream'.
    """
    if task_name.startswith("khona."):
        # Mod-Cog tasks adapter
        return make_modcog_dataset(task_name, batch_size=batch_size, seq_len=seq_len or 350)
    elif task_name.startswith("poli."):
        # Register them first
        from curiosity.poli_tasks import register_poli_tasks
        try:
            register_poli_tasks()
        except:
            pass # Already registered

    # Choose Dataset Class
    if dataset_mode == 'episode':
        return YangEpisodeDataset(task_name, dt=dt, batch_size=batch_size, timing=None,
                                  rule_idx=rule_idx, n_rules=n_rules, rule_vector=rule_vector)
    else:
        # 'stream' or legacy
        return TaskDataset(task_name, dt=dt, batch_size=batch_size, seq_len=seq_len,
                           rule_idx=rule_idx, n_rules=n_rules, rule_vector=rule_vector)

def single_trial_len_steps(env) -> int:
    _ = env.unwrapped.new_trial()
    return env.unwrapped.ob.shape[0]

def estimate_trial_len_steps(task_name: str, dt: float, timing=None, n: int = 64) -> int:
    env_kwargs = {'dt': dt}
    if timing is not None:
        env_kwargs['timing'] = timing
    ds = ngym.Dataset(task_name, env_kwargs=env_kwargs, batch_size=1, seq_len=32)
    env = ds.env
    lens = [single_trial_len_steps(env) for _ in range(n)]
    return int(max(lens))

def sample_batch_with_decision(ds, device, tries: int = 8, sigma_x: float = 0.0, alpha: float = 0.2):
    """
    Supports both TaskDataset (Yang) and ModCogDataset (Khona).
    Now supports Yang-style input noise injection: x += N(0,1) * sigma_x * sqrt(2/alpha).
    """
    import torch
    import math
    
    # Calculate effective noise std dev if sigma_x > 0
    noise_std = 0.0
    if sigma_x > 0 and alpha > 0:
        noise_std = sigma_x * math.sqrt(2.0 / alpha)

    def _process_batch(x_np, y_np):
        x = torch.from_numpy(x_np).float().to(device)
        y_labels = torch.from_numpy(y_np).long().to(device)
        
        # Inject noise if requested
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
            
        return x, y_labels

    for _ in range(tries):
        x_np, y_np = ds()
        # Valid decision frames in both conventions: y ∉ {-1, 0}
        # We check numpy array to avoid unnecessary GPU transfer if we reject
        if ((y_np != -1) & (y_np != 0)).any():
            x, y_labels = _process_batch(x_np, y_np)
            T, B = x.shape[0], x.shape[1]
            return x, y_labels, T, B
            
    # If we failed to find a decision batch, just return the last one
    x, y_labels = _process_batch(x_np, y_np)
    T, B = x.shape[0], x.shape[1]
    return x, y_labels, T, B

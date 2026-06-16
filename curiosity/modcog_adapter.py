# curiosity/modcog_adapter.py
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
import types
import importlib
import sys

# Gym / NeuroGym
import gym
import neurogym as ng

# --- Minimal NeuroGym shim: expose spaces and core at top-level for Mod_Cog ---
try:
    ng_spaces = importlib.import_module("neurogym.utils.spaces")
except Exception:
    # fallback to gym.spaces if NG wrapper isn't present
    from gym import spaces as ng_spaces  # type: ignore
sys.modules["neurogym.spaces"] = ng_spaces
ng.spaces = ng_spaces
# expose core classes if not already present
core = importlib.import_module("neurogym.core")
if not hasattr(ng, "TrialEnv"):
    ng.TrialEnv = core.TrialEnv  # type: ignore[attr-defined]
if not hasattr(ng, "TrialWrapper"):
    ng.TrialWrapper = core.TrialWrapper  # type: ignore[attr-defined]
if not hasattr(ng, "_SB3_INSTALLED"):
    ng._SB3_INSTALLED = False
# -----------------------------------------------------------------------------

# Import Mod_Cog task constructors
from Mod_Cog import mod_cog_tasks as MC  # type: ignore


def _patch_to_4tuple(env: gym.Env) -> gym.Env:
    """Normalize reset/step to (obs, reward, done, info) for older loops."""
    orig_reset, orig_step = env.reset, env.step

    def reset_patched(self, *a, **k):
        out = orig_reset(*a, **k)
        # tolerate gym>=0.26 (obs, info)
        return out[0] if (isinstance(out, tuple) and len(out) == 2) else out

    def step_patched(self, action):
        out = orig_step(action)
        # new gym API: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, r, term, trunc, info = out
            return obs, r, bool(term or trunc), info
        return out
    env.reset = types.MethodType(reset_patched, env)
    env.step  = types.MethodType(step_patched,  env)
    return env


def _get_ctor(modcog_name: str):
    """Return constructor from Mod_Cog by short name, e.g. 'dm1seql'."""
    if not hasattr(MC, modcog_name):
        raise KeyError(f"Mod_Cog task '{modcog_name}' not found in this checkout.")
    fn = getattr(MC, modcog_name)
    if not callable(fn):
        raise KeyError(f"Mod_Cog symbol '{modcog_name}' is not callable.")
    return fn


class ModCogDataset:
    """Simple sampler that mimics NeuroGym Dataset interface used in loops.

    __call__() -> (X, Y): X[T,B,D] float32, Y[T,B] int64 labels.
    env: underlying env for evaluation.
    """
    def __init__(self, env: gym.Env, batch_size: int = 16, seq_len: int = 350):
        self.env = env
        self.B = int(batch_size)
        self.T = int(seq_len)
        # Infer obs dim from a safe first step
        _ = self.env.reset()
        o, _, _, _ = self.env.step(self.env.action_space.sample())
        self.obs_dim = int(o.shape[0])

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        T, B, D = self.T, self.B, self.obs_dim
        X = np.zeros((T, B, D), dtype=np.float32)
        Y = np.full((T, B), fill_value=-1, dtype=np.int64)  # ignore_index by default
        _ = self.env.reset()
        for b in range(B):
            _ = self.env.reset()
            for t in range(T):
                o, r, d, info = self.env.step(self.env.action_space.sample())
                lo = int(getattr(o, "shape", [len(o)])[0])
                if lo != D:
                    arr = np.asarray(o, np.float32)
                    if lo < D:
                        pad = np.zeros((D-lo,), dtype=np.float32)
                        o = np.concatenate([arr, pad], axis=0)
                    else:
                        o = arr[:D]
                X[t, b] = o
                if isinstance(info, dict) and ("gt" in info):
                    gt = info["gt"]
                    try:
                        arr = np.asarray(gt)
                        gt = int(arr.argmax()) if arr.ndim > 0 else int(gt)
                    except Exception:
                        gt = int(gt)
                    Y[t, b] = gt
                if d:
                    _ = self.env.reset()
        return X, Y


def make_modcog_dataset(name: str, *, batch_size: int = 16, seq_len: int = 350) -> ModCogDataset:
    """Create a dataset from a 'khona.<taskname>' or short '<taskname>' string."""
    short = name.split(".", 1)[1] if name.startswith("khona.") else name
    ctor = _get_ctor(short)
    env = ctor()
    env = _patch_to_4tuple(env)
    return ModCogDataset(env, batch_size=batch_size, seq_len=seq_len)

from __future__ import annotations
from typing import List, Protocol, Tuple
import numpy as np

class BanditPolicy(Protocol):
    def sample(self) -> Tuple[int, str, float]: ...
    def update(self, arm_idx: int, p_arm: float, reward: float) -> None: ...
    def scale_reward(self, rhat: float) -> float: ...

class Exp3S:
    def __init__(self, task_names: List[str], eta=1e-3, eps=0.05, share=0.0, reservoir=5000, qlo=0.2, qhi=0.8, seed=0, scale_mode: str = 'quantile'):
        self.tasks = list(task_names)
        self.K = len(self.tasks)
        self.w = np.ones(self.K, dtype=np.float64)
        self.eta = float(eta); self.eps = float(eps); self.share = float(share)
        self.R = []; self.reservoir = int(reservoir)
        self.qlo = float(qlo); self.qhi = float(qhi)
        self.rng = np.random.RandomState(seed)
        self.scale_mode = scale_mode

    def _probs(self) -> np.ndarray:
        wsum = self.w.sum()
        p = (self.w / wsum) if wsum > 0 else (np.ones(self.K) / self.K)
        p = (1 - self.eps) * p + self.eps * (np.ones(self.K) / self.K)
        return p

    def sample(self):
        p = self._probs()
        i = int(self.rng.choice(self.K, p=p))
        return i, self.tasks[i], float(p[i])

    def scale_reward(self, rhat: float) -> float:
        self.R.append(float(rhat))
        if len(self.R) > self.reservoir:
            self.R.pop(0)
        if len(self.R) < 64:
            return float(np.tanh(rhat))
        if self.scale_mode == 'quantile':
            lo = float(np.quantile(self.R, self.qlo))
            hi = float(np.quantile(self.R, self.qhi))
            r = float(np.clip(rhat, lo, hi))
            if hi == lo:
                return 0.0
            return 2.0 * (r - lo) / (hi - lo) - 1.0
        elif self.scale_mode == 'tanh':
            return float(np.tanh(rhat))
        elif self.scale_mode == 'zscore':
            mu = float(np.mean(self.R)); sd = float(np.std(self.R) + 1e-8)
            return float((rhat - mu) / sd)
        else:
            return float(rhat)

    def update(self, arm_idx: int, p_arm: float, reward: float) -> None:
        g = float(reward) / max(float(p_arm), 1e-8)
        self.w[arm_idx] *= np.exp(self.eta * g)
        if self.share > 0:
            mean_w = (self.w.sum() - self.w[arm_idx]) / max(self.K - 1, 1)
            self.w = (1 - self.share) * self.w + self.share * mean_w

"""
Khona et al.–style classification head for NeuroGym/Yang tasks (PyTorch)

Key choices (per Khona et al., 2023 style):
- Ring output trained with Softmax + Cross-Entropy on *integer class labels*.
- Unified output head (fixation as class 0, others as ring locations).
- Proper *masked* normalization: losses are averaged ONLY over valid (unmasked) frames.
- Forward() returns: ring probabilities (including fixation), logits, and hidden states.

Also includes:
- Optional "separate input" path (sensory + rule linear projections) like the TF code path.
- Leaky integration cell with recurrent noise.
- Optional brain constraints wiring kept intact via BrainConstrained base + _get_Wrec_tensor().

Usage expectations (recommended):
- Provide 'y' and 'c_mask' as dicts:
    y = {
        "ring": LongTensor[T, B],        # integer class labels in [0..n_ring-1]; put -100 where invalid
        "fix":  FloatTensor[T, B],       # targets in {0,1} (or [0,1] if smoothed)
    }
    c_mask = {
        "ring": Float/Bool Tensor[T, B], # 1.0 (or weight) where loss applies, 0.0 otherwise
        "fix":  Float/Bool Tensor[T, B],
    }
- If you pass tensors instead (legacy): 
    * If y.shape == (T,B,n_ring+1): interpret y[..., :n_ring] as one-hot ring targets and y[..., -1] as fixation target.
    * If c_mask matches that shape, split the masks accordingly; else broadcast a scalar/2D mask to both.

Author: refactor for CE+masking by ChatGPT (2025-10-22)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional brain constraints (no-ops if the package isn't present at runtime)
try:
    from brain.nn import BrainConstrained
    from brain.masks import load_connectome
    from brain.constraints import MaskConstraint, DalesLawConstraint, SpectralRadiusRescale
except Exception:
    class BrainConstrained(nn.Module):
        def __init__(self): super().__init__()
        def reapply_brain_constraints_(self): return self
    def load_connectome(*args, **kwargs): 
        raise RuntimeError("load_connectome not available")
    class MaskConstraint: 
        def __init__(self, *a, **k): pass
    class DalesLawConstraint: 
        def __init__(self, *a, **k): pass
    class SpectralRadiusRescale:
        def __init__(self, *a, **k): pass
        def reapply_(self, *a, **k): return self

def get_activation_and_scales(name: str):
    name = name.lower()
    if name == 'softplus':
        act = F.softplus; w_in_start = 1.0; w_rec_start = 0.5
    elif name == 'tanh':
        act = torch.tanh;  w_in_start = 1.0; w_rec_start = 1.0
    elif name == 'relu':
        act = F.relu;      w_in_start = 1.0; w_rec_start = 0.5
    elif name == 'power':
        act = lambda x: F.relu(x).pow(2); w_in_start = 1.0; w_rec_start = 0.01
    elif name == 'retanh':
        act = lambda x: torch.tanh(F.relu(x)); w_in_start = 1.414; w_rec_start = 0.5
    else:
        raise ValueError(f"Unknown activation {name}")
    return act, float(w_in_start), float(w_rec_start)

class LeakyRNNCell(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, alpha: float, sigma_rec: float,
                 activation: str = 'softplus', w_rec_init: str = 'diag', 
                 w_rec_noise: float = 0.01, # New param for noise
                 rng: Optional[np.random.RandomState] = None):
        super().__init__()
        act, w_in_start, w_rec_start = get_activation_and_scales(activation)
        self.act = act
        self.alpha = float(alpha)
        self.sigma = math.sqrt(2.0 / self.alpha) * float(sigma_rec)
        self.n_input = n_input; self.n_hidden = n_hidden

        if rng is None:
            rng = np.random.RandomState()
        w_in0 = (rng.randn(n_input, n_hidden) / math.sqrt(n_input)) * w_in_start
        if w_rec_init == 'diag':
            # Identity + Small Random Noise (to break symmetry/allow mixing)
            w_rec0 = np.eye(n_hidden) * w_rec_start
            if w_rec_noise > 0:
                noise = (rng.randn(n_hidden, n_hidden) / math.sqrt(n_hidden)) * w_rec_noise
                w_rec0 = w_rec0 + noise
        elif w_rec_init == 'randortho':
            A = rng.randn(n_hidden, n_hidden); q, _ = np.linalg.qr(A); w_rec0 = q * w_rec_start
        elif w_rec_init == 'randgauss':
            w_rec0 = (rng.randn(n_hidden, n_hidden) / math.sqrt(n_hidden)) * w_rec_start
        else:
            raise ValueError('w_rec_init must be diag|randortho|randgauss')
        W0 = np.concatenate([w_in0, w_rec0], axis=0).astype(np.float32)
        self.W = nn.Parameter(torch.from_numpy(W0))        # [n_input+n_hidden, n_hidden]
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        pre = torch.matmul(torch.cat([x_t, h], dim=-1), self.W) + self.b
        if self.training and self.sigma > 0:
            pre = pre + torch.randn_like(h) * self.sigma
        h_new = self.act(pre)
        return (1 - self.alpha) * h + self.alpha * h_new

class LeakyRNNCellSeparateInput(nn.Module):
    def __init__(self, n_hidden: int, alpha: float, sigma_rec: float,
                 activation: str = 'softplus', w_rec_init: str = 'diag', 
                 w_rec_noise: float = 0.01,
                 rng: Optional[np.random.RandomState] = None):
        super().__init__()
        act, _, w_rec_start = get_activation_and_scales(activation)
        self.act = act
        self.alpha = float(alpha)
        self.sigma = math.sqrt(2.0 / self.alpha) * float(sigma_rec)
        self.n_hidden = n_hidden
        if rng is None:
            rng = np.random.RandomState()
        if w_rec_init == 'diag':
            w_rec0 = np.eye(n_hidden) * w_rec_start
            if w_rec_noise > 0:
                noise = (rng.randn(n_hidden, n_hidden) / math.sqrt(n_hidden)) * w_rec_noise
                w_rec0 = w_rec0 + noise
        elif w_rec_init == 'randortho':
            A = rng.randn(n_hidden, n_hidden); q, _ = np.linalg.qr(A); w_rec0 = q * w_rec_start
        elif w_rec_init == 'randgauss':
            w_rec0 = (rng.randn(n_hidden, n_hidden) / math.sqrt(n_hidden)) * w_rec_start
        else:
            raise ValueError('w_rec_init must be diag|randortho|randgauss')
        self.Wrec = nn.Parameter(torch.from_numpy(w_rec0.astype(np.float32)))
        self.b    = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, input_proj_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        pre = torch.matmul(h, self.Wrec) + input_proj_t + self.b
        if self.training and self.sigma > 0:
            pre = pre + torch.randn_like(h) * self.sigma
        h_new = self.act(pre)
        return (1 - self.alpha) * h + self.alpha * h_new

@dataclass
class HParams:
    # Core sizes
    n_input: int
    n_rnn: int
    n_ring: Optional[int] = None    # preferred name for output size
    n_output: Optional[int] = None  # legacy alias for n_ring

    # Init / Dynamics
    w_rec_init: str = 'diag'
    w_rec_noise: float = 0.0        # Yang uses no noise in initialization (only sigma_rec)
    activation: str = 'softplus'
    dt: float = 20.0
    tau: float = 100.0
    sigma_rec: float = 0.05         # Yang's default: recurrent noise during training

    # Loss/behavior (loss is hardcoded as CE in loops.py)
    use_separate_input: bool = True   # Khona-style separate input is the default
    # To be filled in by run.py based on obs_dim and #tasks
    rule_start: int = 0               # index where rule inputs begin
    n_rule: int = 0                   # number of rule input channels

    # Regularization (activity only - weight reg is in cfg, not hp)
    l1_h: float = 0.0
    l2_h: float = 0.0

    # Optim (only lr is used by make_optimizer)
    learning_rate: float = 1e-3

    # Misc
    seed: int = 0

class Yang19KhonaModel(BrainConstrained):
    """
    Yang-style RNN with Khona-style heads and loss.
    """
    def __init__(self, hp: HParams, mask_path=None, mask_as_init=False, use_dales=False, 
                 exc_frac: float = 0.8, target_sr: Optional[float] = None, dist_path: Optional[str] = None):
        nn.Module.__init__(self)
        BrainConstrained.__init__(self)
        self.hp = hp
        # Resolve n_ring (backward compatibility)
        if self.hp.n_ring is None:
            if self.hp.n_output is None:
                raise ValueError("Please provide HParams.n_ring (preferred) or legacy HParams.n_output.")
            self.hp.n_ring = int(self.hp.n_output)

        torch.manual_seed(hp.seed); np.random.seed(hp.seed)
        self.alpha = float(hp.dt) / float(hp.tau)

        # --- Sanity checks for separate-input layout ---
        if hp.use_separate_input:
            if hp.rule_start <= 0 or hp.n_rule <= 0:
                raise ValueError(
                    f"use_separate_input=True but got rule_start={hp.rule_start}, n_rule={hp.n_rule}. "
                    "These must be > 0 and are expected to be set in run.py based on obs_dim and #tasks."
                )
            if hp.rule_start + hp.n_rule > hp.n_input:
                raise ValueError(
                    f"rule_start + n_rule = {hp.rule_start + hp.n_rule} exceeds n_input={hp.n_input}. "
                    "Check that the last n_rule dims of the observation are the rule one-hot."
                )
                
        # Input path
        if hp.use_separate_input:
            self.sensory_in = nn.Linear(hp.rule_start, hp.n_rnn)
            self.rule_in    = nn.Linear(hp.n_rule,     hp.n_rnn, bias=False)
            
            # Fix: Use the correct gain (w_in_start) for the chosen activation
            _, w_in_start, _ = get_activation_and_scales(hp.activation)
            # Bound for uniform init: gain * sqrt(3 / fan_in)
            # fan_in = hp.rule_start
            bound = w_in_start * math.sqrt(3.0 / hp.rule_start)
            nn.init.uniform_(self.sensory_in.weight, -bound, bound)
            
            nn.init.zeros_(self.sensory_in.bias)
            nn.init.orthogonal_(self.rule_in.weight)
            self.cell_sep = LeakyRNNCellSeparateInput(hp.n_rnn, self.alpha, hp.sigma_rec,
                                                      activation=hp.activation, w_rec_init=hp.w_rec_init,
                                                      w_rec_noise=hp.w_rec_noise, # PASS NOISE
                                                      rng=np.random.RandomState(hp.seed))
            self.cell = None
        else:
            self.cell = LeakyRNNCell(hp.n_input, hp.n_rnn, self.alpha, hp.sigma_rec,
                                     activation=hp.activation, w_rec_init=hp.w_rec_init,
                                     w_rec_noise=hp.w_rec_noise, # PASS NOISE
                                     rng=np.random.RandomState(hp.seed))
            self.cell_sep = None

        # Single categorical head over NeuroGym actions (0 = fixation, >0 = choices)
        self.w_out = nn.Linear(hp.n_rnn, hp.n_ring)
        nn.init.xavier_uniform_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)

        # ---- Optional brain-based constraints
        constraints = []
        H = hp.n_rnn
        get_weight = lambda module: module._get_Wrec_tensor()

        if mask_path is not None:
            mask_np, adj_np = load_connectome(mask_path, target_sr=target_sr)
            if mask_np.shape != (H, H):
                raise ValueError(f"mask shape {mask_np.shape} must match hidden size {(H,H)}")
            self.register_buffer('brain_mask', torch.tensor(mask_np, dtype=torch.float32))
            if mask_as_init:
                with torch.no_grad():
                    Wrec = self._get_Wrec_tensor()
                    Wrec.copy_(torch.tensor(adj_np, dtype=Wrec.dtype, device=Wrec.device))
            constraints.append(MaskConstraint(get_weight, self.brain_mask))

        # NEW: optional distance matrix buffer (H×H)
        self.register_buffer("distance_matrix", None)
        if dist_path is not None:
            try:
                from brain.masks import _load_array  # reuse existing loader
                D = _load_array(dist_path)
            except Exception:
                D = np.load(dist_path)
            H = int(hp.n_rnn)
            if D.shape != (H, H):
                raise ValueError(f"distance matrix must be ({H},{H}); got {D.shape}")
            self.distance_matrix = torch.as_tensor(D, dtype=torch.float32)
            # Normalize distance matrix so max distance is 1.0
            # This makes lambda values comparable to standard L1
            if self.distance_matrix.max() > 0:
                self.distance_matrix = self.distance_matrix / self.distance_matrix.max()

        if use_dales:
            n_exc = int(round(float(exc_frac) * H))
            signs = torch.ones(H, dtype=torch.float32); signs[n_exc:] = -1.0
            self.register_buffer('brain_dale_signs', signs)
            constraints.append(DalesLawConstraint(get_weight, self.brain_dale_signs))

        if target_sr is not None:
            SpectralRadiusRescale(get_weight, target_sr=target_sr).reapply_(self)

        self._brain_constraints = list(constraints)
        if (getattr(self, 'cell', None) is not None or getattr(self, 'cell_sep', None) is not None) and len(self._brain_constraints) > 0:
            self.reapply_brain_constraints_()

    # ---- constraint handle
    def _get_Wrec_tensor(self) -> torch.Tensor:
        if getattr(self, 'cell_sep', None) is not None:
            return self.cell_sep.Wrec
        if getattr(self, 'cell', None) is not None:
            return self.cell.W[self.hp.n_input:, :]
        raise AttributeError('Model has neither cell nor cell_sep initialized')

    # ---- forward
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [T, B, n_input]
        Returns dict with:
            - ring_logits: [T, B, n_ring]
            - ring_probs:  [T, B, n_ring] (softmax)
            - fix_logits:  [T, B, 1]
            - fix_prob:    [T, B, 1] (sigmoid)
            - h:           [T, B, n_rnn]
        """
        hp = self.hp
        T, B, _ = x.shape
        h = torch.zeros(B, hp.n_rnn, device=x.device, dtype=x.dtype)
        hs = []
        for t in range(T):
            x_t = x[t]
            if hp.use_separate_input:
                inp = self.sensory_in(x_t[..., :hp.rule_start])
                rule_proj = self.rule_in(x_t[..., hp.rule_start:hp.rule_start+hp.n_rule])
                h = self.cell_sep(inp + rule_proj, h)
            else:
                h = self.cell(x_t, h)
            hs.append(h)
        h_seq = torch.stack(hs, dim=0)
        logits = self.w_out(h_seq)                # (T,B,n_actions)
        probs  = torch.softmax(logits, dim=-1)
        return {"ring_logits": logits, "ring_probs": probs, "h": h_seq}


    # ---- loss (Khona CE style)
    @staticmethod
    def _to_bool_mask(m: torch.Tensor) -> torch.Tensor:
        if m.dtype == torch.bool:
            return m
        return m != 0


def make_optimizer(params, hp: HParams):
    # Optimizer is always Adam (hardcoded, not in hp)
    return torch.optim.Adam(params, lr=hp.learning_rate)

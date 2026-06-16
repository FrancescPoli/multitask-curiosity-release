
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import torch.nn as nn

# A function taking (module) and returning a Tensor reference to the recurrent weight (view or param)
WeightGetter = Callable[[nn.Module], torch.Tensor]

class Constraint:
    def reapply_(self, module: nn.Module) -> None:
        raise NotImplementedError

@dataclass
class MaskConstraint(Constraint):
    get_weight: WeightGetter
    mask: torch.Tensor  # [H,H] float tensor 0/1
    def reapply_(self, module: nn.Module) -> None:
        W = self.get_weight(module)
        M = self.mask
        if isinstance(M, torch.Tensor):
            # ensure mask lives with W
            if M.device != W.device or M.dtype != W.dtype:
                M = M.to(device=W.device, dtype=W.dtype)
        else:
            # allow numpy/array-like masks
            M = torch.as_tensor(M, device=W.device, dtype=W.dtype)

        with torch.no_grad():
            W.mul_(M)

@dataclass
class DalesLawConstraint(Constraint):
    get_weight: WeightGetter
    signs: torch.Tensor  # [+1 or -1] shape [H]
    def reapply_(self, module: nn.Module) -> None:
        W = self.get_weight(module)
        s = self.signs.to(W.device, dtype=W.dtype).view(-1, 1)  # row-wise sign
        with torch.no_grad():
            W.copy_(s * W.abs())

@dataclass
class SpectralRadiusRescale(Constraint):
    get_weight: WeightGetter
    target_sr: Optional[float] = 0.9
    eps: float = 1e-8
    iters: int = 20
    def reapply_(self, module: nn.Module) -> None:
        if self.target_sr is None: return
        W = self.get_weight(module)
        with torch.no_grad():
            # power iteration on W^T W
            v = torch.randn(W.shape[0], 1, device=W.device, dtype=W.dtype)
            for _ in range(self.iters):
                v = (W.T @ (W @ v))
                n = torch.linalg.norm(v) + self.eps
                v = v / n
            # Rayleigh quotient gives spectral radius squared of W
            r2 = (v.T @ (W.T @ (W @ v))).squeeze()
            sr = torch.sqrt(torch.clamp(r2, min=self.eps)).item()
            if sr > 0:
                W.mul_(float(self.target_sr) / sr)

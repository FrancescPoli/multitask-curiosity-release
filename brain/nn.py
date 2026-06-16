
from __future__ import annotations
from typing import List
import torch.nn as nn
from .constraints import Constraint

class BrainConstrained(nn.Module):
    def __init__(self, *constraints: Constraint):
        # NOTE: do NOT call super().__init__() here; let the concrete nn.Module subclass
        # call nn.Module.__init__(self) exactly once to avoid resetting registered submodules.
        self._brain_constraints: List[Constraint] = list(constraints)

    def reapply_brain_constraints_(self) -> None:
        for c in self._brain_constraints:
            c.reapply_(self)

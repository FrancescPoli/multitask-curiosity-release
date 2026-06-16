
"""
Foraging-based curriculum controller (MVT / Constantino & Daw style).

This module implements a *task-agnostic* stay/leave policy driven by
learning progress. It does NOT know about RNNs, losses, or datasets;
it only consumes a scalar "reward" per step (e.g., Δloss) and decides
whether to stay on the current task or leave and incur a travel cost.

Intended usage (inside a training loop):

    from curiosity.train.foraging import MVTCurriculumController

    forager = MVTCurriculumController(...)
    current_task = initial_task
    forager.start_new_task(current_task)

    for global_step in range(cfg.steps):
        if forager.in_travel():
            info = forager.step_travel()   # reward=0, updates rho
            if forager.travel_remaining == 0:
                current_task = pick_new_task(...)
                forager.start_new_task(current_task)
            continue

        # 1) Run one gradient step on current_task, obtain scalar reward_t
        info = forager.step_training(reward_t)
        if info["action"] == 1:  # leave
            # next loop iteration will be travel; keep current_task as is
            pass

The key variables:
    - rho_t: EMA of reward per *time step* (including travel zeros)
    - P_t:   EMA of reward on the current task within the current block
    - min_block_steps: minimum steps before we allow leaving
    - travel_steps:    number of zero-reward steps when leaving

The MVT-inspired decision rule:
    - If block_steps < min_block_steps: forced stay.
    - Else: stay if P_t > rho_t + eps, otherwise leave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ForagingState:
    """Lightweight container for introspection / logging."""
    rho: float
    P: float
    current_task: Optional[str]
    block_steps: int
    travel_remaining: int


class MVTCurriculumController:
    """Average-reward stay/leave controller for task foraging.

    Parameters
    ----------
    alpha_local : float
        EMA rate for local progress P_t on the *current task*.
        Larger values → P_t reacts more quickly to new rewards.
    beta_global : float
        EMA rate for the global baseline rho_t (average reward per step,
        including travel zeros). Smaller values → rho_t changes slowly.
    min_block_steps : int
        Minimum number of training steps on a task before leaving can
        be considered (i.e., before we compare P_t vs rho_t + eps).
    travel_steps : int
        Number of zero-reward time steps when leaving a task (travel cost).
    eps : float
        Margin in the MVT comparison. We stay if

            P_t > rho_t + eps

        and leave otherwise. Positive eps makes leaving easier; negative
        eps makes leaving harder.
    """

    def __init__(
        self,
        alpha_local: float = 0.03,
        beta_global: float = 0.003,
        min_block_steps: int = 10,
        travel_steps: int = 50,
        eps: float = 0.0,
        temperature: float = 0.0,
    ) -> None:
        # Hyperparameters
        self.alpha_local = float(alpha_local)
        self.beta_global = float(beta_global)
        self.min_block_steps = int(min_block_steps)
        self.travel_steps = int(travel_steps)
        self.eps = float(eps)
        self.temperature = float(temperature)

        # State
        self.rho: float = 0.0
        self.P: float = 0.0
        self.current_task: Optional[str] = None
        self.block_steps: int = 0
        self.travel_remaining: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_rho(self, r_t: float) -> None:
        """Global baseline update: rho <- (1 - beta) * rho + beta * r_t."""
        beta = self.beta_global
        self.rho = (1.0 - beta) * self.rho + beta * float(r_t)

    def _update_P(self, r_t: float) -> None:
        """Local progress update: P <- (1 - alpha) * P + alpha * r_t."""
        alpha = self.alpha_local
        self.P = (1.0 - alpha) * self.P + alpha * float(r_t)

    # ------------------------------------------------------------------
    # Public API used by training code
    # ------------------------------------------------------------------

    def get_state(self) -> ForagingState:
        """Return a snapshot of the internal state (for logging/debug)."""
        return ForagingState(
            rho=self.rho,
            P=self.P,
            current_task=self.current_task,
            block_steps=self.block_steps,
            travel_remaining=self.travel_remaining,
        )

    # ---- task / travel management ------------------------------------

    def start_new_task(self, task_name: str) -> None:
        """Call when we *enter* a new task after travel.

        This resets the local progress P and block step counter, but leaves
        the global baseline rho untouched.
        """
        self.current_task = str(task_name)
        self.P = 0.0
        self.block_steps = 0

    def in_travel(self) -> bool:
        """Return True if we are currently in travel (i.e., not training)."""
        return self.travel_remaining > 0

    # ---- one step of travel ------------------------------------------

    def step_travel(self) -> Dict[str, Any]:
        """Advance the controller by one *travel* time step.

        Travel is a period of zero reward (r_t = 0) where we still update
        rho_t (so travel time lowers the average reward rate).

        Returns a dict with:
            mode:            'travel'
            action:          -1  (no stay/leave decision)
            rho:             current rho_t after update
            P:               current P_t (unchanged during travel)
            block_steps:     current block length on the previous task
            travel_remaining:number of remaining travel steps after this one
            current_task:    name of the last task (for logging only)
            reward:          0.0
        """
        if self.travel_remaining <= 0:
            raise RuntimeError("step_travel() called but travel_remaining == 0")

        r_t = 0.0
        self._update_rho(r_t)
        self.travel_remaining -= 1

        return {
            "mode": "travel",
            "action": -1,
            "rho": self.rho,
            "P": self.P,
            "block_steps": self.block_steps,
            "travel_remaining": self.travel_remaining,
            "current_task": self.current_task,
            "reward": r_t,
        }

    # ---- one training step on current task ---------------------------

    def step_training(self, reward: float) -> Dict[str, Any]:
        """Advance the controller by one *training* step on the current task.

        Parameters
        ----------
        reward : float
            Scalar reward r_t for this training step, typically defined as
            learning progress, e.g. Δloss = loss_prev - loss_curr.

        Returns
        -------
        info : dict
            A dictionary with:
                mode:          'train'
                action:        0 (stay) or 1 (leave)
                rho:           new rho_t
                P:             new P_t
                block_steps:   length of current block (after this step)
                travel_remaining: travel steps remaining (if any)
                current_task:  task name
                reward:        r_t
                note:          textual explanation ('stay_forced_min_block',
                                 'stay_P>rho', 'leave_P<=rho')
        """
        if self.current_task is None:
            raise RuntimeError("step_training() called before start_new_task().")

        r_t = float(reward)
        # Update global and local averages
        self._update_rho(r_t)
        self._update_P(r_t)
        self.block_steps += 1

        # Default: stay
        action = 0
        note = "stay_forced_min_block"

        # Enforce minimum block length
        if self.block_steps < self.min_block_steps:
            # Do nothing more; forced stay
            pass
        else:
            # MVT-style comparison
            delta = self.P - (self.rho + self.eps)

            # Stochastic decision if temperature > 0
            if self.temperature > 1e-9:
                import numpy as np
                # sigmoid(delta / T)
                # Cap exponent for stability
                val = delta / self.temperature
                val = max(min(val, 20.0), -20.0)
                prob_stay = 1.0 / (1.0 + np.exp(-val))
                
                should_stay = (np.random.rand() < prob_stay)
                if should_stay:
                    action = 0
                    note = f"stay_stoch_p={prob_stay:.2f}"
                else:
                    action = 1
                    note = f"leave_stoch_p={prob_stay:.2f}"
                    self.travel_remaining = self.travel_steps
            else:
                # Deterministic check
                if delta > 0:
                    action = 0
                    note = "stay_P>rho"
                else:
                    action = 1
                    note = "leave_P<=rho"
                    # Leaving: start travel on subsequent steps
                    self.travel_remaining = self.travel_steps

        return {
            "mode": "train",
            "action": action,
            "rho": self.rho,
            "P": self.P,
            "block_steps": self.block_steps,
            "travel_remaining": self.travel_remaining,
            "current_task": self.current_task,
            "reward": r_t,
            "note": note,
        }

#!/usr/bin/env python
"""
Extract fundamental structural features for Mod-Cog tasks.

Fundamental = factual properties of:
  - the trial timeline (which periods exist),
  - integration/sequence transforms,
  - anti vs non-anti mapping,
  - multi-modality structure (single/context/multi-sensory),
  - match vs non-match rules,
  - whether stimuli are present during the decision epoch (stim_in_decision).

Metadata (non-fundamental) are:
  - name   (task constructor name)
  - family (core env class name)

Output: modcog_fundamental_struct.csv
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import math
import inspect

import numpy as np
import pandas as pd

# Optional: adapter
try:
    from curiosity import modcog_adapter as mca  # type: ignore
    HAS_ADAPTER = True
except Exception:
    mca = None
    HAS_ADAPTER = False

# Core Mod-Cog tasks
from Mod_Cog import mod_cog_tasks as MC  # type: ignore

# Base env classes
from Mod_Cog.mod_cog_tasks import (  # type: ignore
    _Reach,
    _DMFamily,
    _DelayMatch1DResponse,
)

from neurogym.wrappers.block import ScheduleEnvs  # type: ignore
from neurogym.core import TrialWrapper  # type: ignore


# ---------------------------------------------------------------------
# Helpers: enumerate task constructors
# ---------------------------------------------------------------------


def list_task_functions(module) -> List[Tuple[str, Any]]:
    """Return list of (name, function) for public Mod-Cog task constructors."""
    funcs: List[Tuple[str, Any]] = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue  # skip helpers: _reach, _dm_kwargs, _dlymatch, ...
        if obj.__module__ != module.__name__:
            continue  # skip imported/re-exported stuff
        funcs.append((name, obj))
    return funcs


# ---------------------------------------------------------------------
# Helpers: unwrap ScheduleEnvs + TrialWrapper to core env
# ---------------------------------------------------------------------


def unwrap_env(env):
    """
    Unwrap env into (schedule_env, wrapper_env, core_env).

    schedule_env: outer ScheduleEnvs (if any)
    wrapper_env:  inner TrialWrapper (e.g. _MultiModalityStimulus) (if any)
    core_env:     underlying _Reach / _DMFamily / _DelayMatch1DResponse
    """
    schedule_env = None
    wrapper_env = None
    core_env = env

    if isinstance(core_env, ScheduleEnvs):
        schedule_env = core_env
        core_env = schedule_env.envs[0]  # inspect first modality/env

    if isinstance(core_env, TrialWrapper):
        wrapper_env = core_env
        core_env = wrapper_env.env

    return schedule_env, wrapper_env, core_env


# ---------------------------------------------------------------------
# Helpers: canonical periods → structural flags
# ---------------------------------------------------------------------


def canonical_periods(core_env, base_class: str, task_name: str) -> list[str]:
    reaction = bool(getattr(core_env, "reaction", False))
    delaycomparison = bool(getattr(core_env, "delaycomparison", False))
    timing = getattr(core_env, "timing", {})

    if isinstance(core_env, _Reach):
        if reaction:
            return ["fixation", "decision"]
        else:
            # --- NEW: check integrate/dly flags, not just static timing ---
            integrate_raw = getattr(core_env, "integrate", 0)
            dly_flag = bool(getattr(core_env, "dly", False))

            delay = timing.get("delay", 0)
            if callable(delay):
                try:
                    delay_val = float(delay())
                except Exception:
                    delay_val = 0.0
            else:
                try:
                    delay_val = float(delay)
                except Exception:
                    delay_val = 0.0

            # If the code will introduce a delay via integrate or dly,
            # treat this task as having a delay structurally.
            if integrate_raw != 0 or dly_flag:
                delay_val = max(delay_val, 1.0)

            if delay_val > 0:
                return ["fixation", "stimulus", "delay", "decision"]
            else:
                return ["fixation", "stimulus", "decision"]

    if isinstance(core_env, _DMFamily):
        if delaycomparison:
            return ["fixation", "stim1", "delay", "stim2", "decision"]
        else:
            return ["fixation", "stimulus", "decision"]

    if isinstance(core_env, _DelayMatch1DResponse):
        return ["fixation", "sample", "delay", "test", "decision"]

    return []



def periods_to_flags(periods: list[str]) -> Dict[str, int]:
    """Map period names to 5 structural booleans."""
    has_fixation = int("fixation" in periods)
    has_decision = int("decision" in periods)

    has_stim1 = int(
        ("stimulus" in periods) or
        ("stim1" in periods) or
        ("sample" in periods)
    )

    has_delay = int("delay" in periods)

    has_stim2 = int(
        ("stim2" in periods) or
        ("test" in periods)
    )

    return {
        "has_fixation": has_fixation,
        "has_stim1": has_stim1,
        "has_delay": has_delay,
        "has_stim2": has_stim2,
        "has_decision": has_decision,
    }


# ---------------------------------------------------------------------
# Core: extract structural features
# ---------------------------------------------------------------------


def extract_fundamental_struct() -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []

    for name, func in list_task_functions(MC):
        # Instantiate env
        try:
            if HAS_ADAPTER:
                ctor = mca._get_ctor(name)  # type: ignore[attr-defined]
                env = ctor()
            else:
                env = func()
        except Exception as e:
            print(f"[WARN] Could not instantiate task '{name}': {e}")
            continue

        _, _, core_env = unwrap_env(env)
        base_class = type(core_env).__name__

        # -------------------------------------------------------------
        # Metadata (non-fundamental)
        # -------------------------------------------------------------
        row: Dict[str, Any] = {
            "name": name,
            "family": base_class,
        }

        # -------------------------------------------------------------
        # Fundamental: integration / sequence
        # -------------------------------------------------------------
        integrate_raw = getattr(core_env, "integrate", 0)
        seq_raw = getattr(core_env, "seq", 0)

        integration = 1 if integrate_raw != 0 else 0
        sequence = 1 if seq_raw != 0 else 0

        if integrate_raw == 0:
            integration_type = "none"
        elif integrate_raw < 0:
            integration_type = "counterclock"
        else:
            integration_type = "clock"

        if seq_raw == 0:
            sequence_type = "none"
        elif seq_raw < 0:
            sequence_type = "counterclock"
        else:
            sequence_type = "clock"

        row["integration"] = integration
        row["integration_type"] = integration_type
        row["sequence"] = sequence
        row["sequence_type"] = sequence_type

        # -------------------------------------------------------------
        # Fundamental: anti (for all tasks)
        # -------------------------------------------------------------
        if isinstance(core_env, _Reach):
            anti = bool(core_env.anti)
        else:
            anti = False
        row["anti"] = anti

        # -------------------------------------------------------------
        # Fundamental: period structure
        # -------------------------------------------------------------
        periods = canonical_periods(core_env, base_class, name)
        row.update(periods_to_flags(periods))

        # -------------------------------------------------------------
        # Fundamental: stim_in_decision
        # -------------------------------------------------------------
        # Only RT reach tasks present the sensory stimulus during decision.
        if isinstance(core_env, _Reach) and bool(getattr(core_env, "reaction", False)):
            row["stim_in_decision"] = 1
        else:
            row["stim_in_decision"] = 0

        # -------------------------------------------------------------
        # Fundamental: decision timing (used in training)
        # -------------------------------------------------------------
        timing = getattr(core_env, "timing", {})
        decision_val = timing.get("decision", None)
        if decision_val is not None and not callable(decision_val):
            try:
                row["timing_decision_ms"] = float(decision_val)
            except Exception:
                row["timing_decision_ms"] = math.nan
        else:
            row["timing_decision_ms"] = math.nan

        # -------------------------------------------------------------
        # Coherence presence (DM family)
        # -------------------------------------------------------------
        row["has_cohs"] = int(isinstance(core_env, _DMFamily))

        # -------------------------------------------------------------
        # Modality structure: has_modality1/2 + w_mod, w_mod_type
        # -------------------------------------------------------------
        if isinstance(core_env, _DMFamily):
            stim_mod1 = bool(getattr(core_env, "stim_mod1", False))
            stim_mod2 = bool(getattr(core_env, "stim_mod2", False))
            w_mod1 = float(getattr(core_env, "w_mod1", 0.0))
            w_mod2 = float(getattr(core_env, "w_mod2", 0.0))
        else:
            stim_mod1 = True
            stim_mod2 = False
            w_mod1 = 1.0
            w_mod2 = 0.0

        row["has_modality1"] = int(stim_mod1)
        row["has_modality2"] = int(stim_mod2)

        if isinstance(core_env, _DMFamily):
            row["w_mod"] = 1
            if stim_mod1 and not stim_mod2:
                w_mod_type = "single_mod"
            elif not stim_mod1 and stim_mod2:
                w_mod_type = "single_mod"
            elif stim_mod1 and stim_mod2:
                if w_mod1 == 1.0 and w_mod2 == 0.0:
                    w_mod_type = "context_mod1"
                elif w_mod1 == 0.0 and w_mod2 == 1.0:
                    w_mod_type = "context_mod2"
                elif w_mod1 == 1.0 and w_mod2 == 1.0:
                    w_mod_type = "multi_sum"
                else:
                    w_mod_type = "other"
            else:
                w_mod_type = "none"
        else:
            row["w_mod"] = 0
            w_mod_type = "none"

        row["w_mod_type"] = w_mod_type

        # -------------------------------------------------------------
        # Match structure (DelayMatch family)
        # -------------------------------------------------------------
        if isinstance(core_env, _DelayMatch1DResponse):
            matchto = getattr(core_env, "matchto", None)
            matchgo = getattr(core_env, "matchgo", None)

            row["has_match_rule"] = 1
            if matchto == "sample":
                row["match_rule_type"] = "sample"
            elif matchto == "category":
                row["match_rule_type"] = "category"
            else:
                row["match_rule_type"] = "other"

            row["has_matchgo"] = 1
            if matchgo is True:
                row["matchgo_type"] = "match"
            elif matchgo is False:
                row["matchgo_type"] = "non_match"
            else:
                row["matchgo_type"] = "other"
        else:
            row["has_match_rule"] = 0
            row["match_rule_type"] = "none"
            row["has_matchgo"] = 0
            row["matchgo_type"] = "none"

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    df = extract_fundamental_struct()
    print(df.head())
    print()
    print(f"Extracted features for {len(df)} Mod-Cog tasks.")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    out_path = "analysis/Taskonomy/Data/modcog_fundamental_struct2.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote fundamental structural feature table to: {out_path}")


if __name__ == "__main__":
    main()



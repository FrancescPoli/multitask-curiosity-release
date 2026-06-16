
"""
Scaffolding Curriculum (Complexity-Weighted Diet)

Restricts the set of available tasks based on their difficulty (compositional complexity).
Unlocks harder levels over time, with duration scaling by difficulty.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

class ScaffoldCurriculum:
    def __init__(
        self,
        tasks: List[str],
        total_steps: int,
        factor: float = 1.0,
        mode: str = 'cumulative',
        num_groups: int = 5,
        csv_path: str = "analysis/Taskonomy/Data/modcog_fundamental_struct2.csv",
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Args:
            tasks: List of task names available in the run.
            total_steps: Total training steps (schedule will be partitioned from this).
            factor: Multiplier for duration of subsequent groups.
            mode: 'cumulative' (tasks accumulate) or 'disjoint' (only current group active).
            num_groups: Number of balanced complexity groups (bins).
            csv_path: Path to the structural feature CSV.
            seed: Random seed for reproducible tie-breaking.
        """
        self.all_tasks = sorted(list(set(tasks)))
        self.total_steps = total_steps
        self.factor = factor
        self.cumulative = (mode == 'cumulative')
        self.num_groups = num_groups
        self.seed = seed
        self.verbose = verbose
        
        # 1. Load Difficulty
        self.task_difficulty: Dict[str, int] = {}
        self._load_difficulty(csv_path)
        
        # 2. Form Groups (Balanced Bins)
        # self.level_tasks now maps Group Index (0..K-1) -> List[Tasks]
        self.level_tasks: Dict[int, List[str]] = {}
        self._form_balanced_groups()
        
        # 3. Compute Schedule (derived from total_steps)
        # schedule[G] = (start_step, end_step)
        self.schedule: Dict[int, tuple[int, int]] = {} 
        self._compute_schedule()
        
        if self.verbose:
            print(f"[Scaffold] Initialized with {len(self.all_tasks)} tasks partitioned into {len(self.level_tasks)} balanced groups.")
            print(f"[Scaffold] Mode: {'Cumulative' if self.cumulative else 'Disjoint'}")
            for grp in sorted(self.level_tasks.keys()):
                start, end = self.schedule[grp]
                tasks_g = self.level_tasks[grp]
                avg_diff = np.mean([self.task_difficulty[t] for t in tasks_g]) if tasks_g else 0
                print(f"  Group {grp} (Avg Diff {avg_diff:.1f}): {start:,} -> {end:,} steps. ({len(tasks_g)} tasks)")

    def _load_difficulty(self, csv_path: str):
        if not Path(csv_path).exists():
            print(f"[WARN] Scaffolding CSV not found at {csv_path}. Difficulty = 0 for all.")
            for t in self.all_tasks:
                self.task_difficulty[t] = 0
            return

        df = pd.read_csv(csv_path)
        
        # --- Feature Engineering (Copied from plot_taskonomy_separate.py) ---
        if "w_mod_type" not in df.columns:
            # Fallback if CSV is old/different schema
            pass 
        else:
            df["is_context_mod"] = df["w_mod_type"].isin(["context_mod1", "context_mod2"]).astype(int)
            df["is_multi_mod"] = (df["w_mod_type"] == "multi_sum").astype(int)

        if "match_rule_type" in df.columns:
            df["is_category_rule"] = (df["match_rule_type"] == "category").astype(int)
        else:
            df["is_category_rule"] = 0
            
        if "matchgo_type" in df.columns:
            df["is_nonmatch_go"] = (df["matchgo_type"] == "non_match").astype(int)
        else:
            df["is_nonmatch_go"] = 0

        # Two-stimulus / comparison primitive:
        cols_needed = ["has_cohs", "has_stim2", "has_match_rule"]
        for c in cols_needed:
            if c not in df.columns: df[c] = 0
            
        df["two_stim_comp"] = (
            (df["has_cohs"] == 1) |
            ((df["has_stim2"] == 1) & (df["has_match_rule"] == 1))
        ).astype(int)

        # Compositional primitives
        comp_cols = [
            "integration",
            "sequence",
            "anti",
            "has_delay",
            "stim_in_decision",
            "is_context_mod",
            "is_multi_mod",
            "is_category_rule",
            "is_nonmatch_go",
            "two_stim_comp",
        ]
        
        # Ensure all exist
        for c in comp_cols:
            if c not in df.columns: df[c] = 0
            
        # Calc comp_count
        df[comp_cols] = df[comp_cols].fillna(0).astype(int)
        df["comp_count"] = df[comp_cols].sum(axis=1)
        # -------------------------------------------------------------------
        
        # Build mapping: short_name -> full_task_id
        # E.g., {'go': 'khona.go', 'rtgo': 'yang19.rtgo-v0'}
        def _strip_prefix(task_id: str) -> str:
            """Strip 'khona.', 'yang19.' prefix and '-v0' suffix."""
            short = task_id
            for prefix in ('khona.', 'yang19.'):
                if short.startswith(prefix):
                    short = short[len(prefix):]
                    break
            if short.endswith('-v0'):
                short = short[:-3]
            return short
        
        short_to_full = {_strip_prefix(t): t for t in self.all_tasks}
        short_names = set(short_to_full.keys())
        
        # Filter DF to relevant tasks (using short names)
        df = df[df['name'].isin(short_names)]
        
        known_tasks = set()
        for _, row in df.iterrows():
            short_name = row['name']
            diff = int(row['comp_count'])
            full_name = short_to_full.get(short_name, short_name)
            self.task_difficulty[full_name] = diff
            known_tasks.add(full_name)
            
        # Check for missing
        missing = set(self.all_tasks) - known_tasks
        if missing:
            print(f"[WARN] Tasks missing from scaffolding CSV: {missing}. Assigning Difficulty 0.")
            for t in missing:
                self.task_difficulty[t] = 0

    def _form_balanced_groups(self):
        """
        Sort tasks by (Difficulty, Random) and split into num_groups.
        """
        rng = np.random.RandomState(self.seed)
        
        # 1. Create list of (task, difficulty, random_val)
        # We assign a random float to each task for stable tie-breaking
        annotated = []
        for t in self.all_tasks:
            diff = self.task_difficulty[t]
            rand_val = rng.rand()
            annotated.append((t, diff, rand_val))
            
        # 2. Sort by Difficulty (asc), then Random (asc)
        # Python sort is stable, but tuple comparison works directly
        annotated.sort(key=lambda x: (x[1], x[2]))
        
        sorted_tasks = [x[0] for x in annotated]
        
        # 3. Split into groups
        # If seed is fixed, this is reproducible.
        n_total = len(sorted_tasks)
        if n_total == 0:
            return
            
        # Use simple splitting
        # If tasks don't divide evenly, earlier groups get +1 or strictly equal?
        # np.array_split handles this well
        if self.num_groups <= 0: self.num_groups = 1 # safety
        
        subgroups = np.array_split(sorted_tasks, self.num_groups)
        
        for g_idx, group in enumerate(subgroups):
            self.level_tasks[g_idx] = list(group)
            
        self.sorted_groups = sorted(self.level_tasks.keys())


    def _compute_schedule(self):
        """
        Derive schedule from total_steps.
        Duration(i) = base * (factor ^ i)
        where base = total_steps / sum(factor^i for i in 0..num_groups-1)
        """
        n = len(self.sorted_groups)
        if n == 0:
            return
        
        # Geometric series sum: S = (1 + r + r^2 + ... + r^(n-1))
        if self.factor == 1.0:
            series_sum = float(n)
        else:
            series_sum = (1.0 - self.factor ** n) / (1.0 - self.factor)
        
        base_steps = self.total_steps / series_sum
        
        current_step = 0
        for i, grp in enumerate(self.sorted_groups):
            duration = int(round(base_steps * (self.factor ** i)))
            
            start = current_step
            end = start + duration
            
            self.schedule[grp] = (start, end)
            current_step = end
            
    def get_active_tasks(self, global_step: int) -> List[str]:
        """
        Return list of allowed tasks for this global step.
        """
        active = []
        
        # Check simple infinite extension of last group
        last_grp = self.sorted_groups[-1]
        last_end = self.schedule[last_grp][1]
        
        if global_step >= last_end:
            return list(self.all_tasks)

        # Normal logic
        current_grp_idx = len(self.sorted_groups) - 1
        for i, grp in enumerate(self.sorted_groups):
            start, end = self.schedule[grp]
            if start <= global_step < end:
                current_grp_idx = i
                break

        # Now gather tasks based on mode
        for i, grp in enumerate(self.sorted_groups):
            if self.cumulative:
                # Active if group <= current_phase
                if i <= current_grp_idx:
                    active.extend(self.level_tasks[grp])
            else:
                # Disjoint: Active only if group == current_phase
                if i == current_grp_idx:
                    active.extend(self.level_tasks[grp])
                    
        return sorted(list(set(active)))

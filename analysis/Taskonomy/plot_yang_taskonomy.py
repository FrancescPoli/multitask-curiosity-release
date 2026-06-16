#!/usr/bin/env python
"""
Yang19-specific state-level compositionality graphs.
Filters modcog_fundamental_struct2.csv to only the 20 tasks used in Yang et al. (2019).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# ---------------------------------------------------------------------
# 0. Define Yang19 Task List
# ---------------------------------------------------------------------
YANG_TASKS = {
    'go', 'rtgo', 'anti', 'rtanti',
    'dm1', 'dm2', 'ctxdm1', 'ctxdm2', 'multidm',
    'dlygo', 'dlyanti',
    'dlydm1', 'dlydm2', 'ctxdlydm1', 'ctxdlydm2', 'multidlydm',
    'dms', 'dnms', 'dmc', 'dnmc'
}

# ---------------------------------------------------------------------
# 1. Load CSV and derive extra primitives
# ---------------------------------------------------------------------

df = pd.read_csv("analysis/Taskonomy/Data/modcog_fundamental_struct2.csv")

# FILTER FOR YANG TASKS
df = df[df["name"].isin(YANG_TASKS)].copy()
print(f"Filtered to {len(df)} Yang19 tasks.")

# Derive modality-role primitives from w_mod_type
if "w_mod_type" not in df.columns:
    raise ValueError("Expected column 'w_mod_type' not found in modcog_fundamental_struct2.csv")

df["is_context_mod"] = df["w_mod_type"].isin(["context_mod1", "context_mod2"]).astype(int)
df["is_multi_mod"] = (df["w_mod_type"] == "multi_sum").astype(int)

# Derive match-rule primitives (DelayMatch family)
df["is_category_rule"] = (df["match_rule_type"] == "category").astype(int)
df["is_nonmatch_go"] = (df["matchgo_type"] == "non_match").astype(int)

# Two-stimulus / comparison primitive:
# DM: coherence-based comparison (has_cohs == 1)
# DelayMatch: sample–test comparison (has_stim2 & has_match_rule)
df["two_stim_comp"] = (
    (df["has_cohs"] == 1) |
    ((df["has_stim2"] == 1) & (df["has_match_rule"] == 1))
).astype(int)

# Compositional primitives
# Removed 'integration' and 'sequence' as they are 0 for all Yang tasks (mostly)
# Re-evaluating: Does Yang have ANY int/seq? Checking CSV..
# Assuming they are base tasks, int/seq columns should be 0.
# We keep the columns to stay consistent, but they might be empty.
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

# Ensure numeric 0/1 where relevant
df[comp_cols] = df[comp_cols].fillna(0).astype(int)

# Compositional count (for vertical layering)
df["comp_count"] = df[comp_cols].sum(axis=1)

# ---------------------------------------------------------------------
# 2. Build STATES: (family, comp_vector)
# ---------------------------------------------------------------------

state_groups = defaultdict(list)

for idx, row in df.iterrows():
    fam = row["family"]
    comp_vec = tuple(int(row[c]) for c in comp_cols)
    key = (fam, comp_vec)
    state_groups[key].append(idx)

state_keys = list(state_groups.keys())
n_states = len(state_keys)
print(f"Found {n_states} distinct compositional states.")

# For each state: comp vector, comp_count, tasks, family
state_comp_vectors = []
state_comp_counts = []
state_tasks = []
state_family = []

for fam, vec in state_keys:
    idxs = state_groups[(fam, vec)]
    state_comp_vectors.append(np.array(vec, dtype=int))
    state_comp_counts.append(sum(vec))
    task_names = [df.loc[i, "name"] for i in idxs]
    state_tasks.append(task_names)
    state_family.append(fam)

if n_states > 0:
    state_comp_vectors = np.stack(state_comp_vectors, axis=0)
    state_comp_counts = np.array(state_comp_counts, dtype=int)
else:
    print("No states found! Check filtering.")
    exit()

# ---------------------------------------------------------------------
# 3. Build compositional edges between STATES
# ---------------------------------------------------------------------

def build_edges_states_strict():
    edges = []
    for i in range(n_states):
        for j in range(n_states):
            if i == j: continue
            if state_family[i] != state_family[j]: continue
            vi = state_comp_vectors[i]
            vj = state_comp_vectors[j]
            if not (np.all(vi <= vj) and np.any(vi < vj)): continue
            if state_comp_counts[j] != state_comp_counts[i] + 1: continue
            diff_idx = np.where(vi != vj)[0]
            if len(diff_idx) != 1: continue
            feat_name = comp_cols[diff_idx[0]]
            edges.append((i, j, feat_name))
    return edges

edges_states_strict = build_edges_states_strict()
print(f"Strict state graph: {len(edges_states_strict)} edges.")

# ---------------------------------------------------------------------
# 4. Layout
# ---------------------------------------------------------------------

levels = sorted(set(state_comp_counts.tolist()))
families_unique = ["_Reach", "_DMFamily", "_DelayMatch1DResponse"]
fam_offset = {
    "_Reach": -10.0,
    "_DMFamily": 0.0,
    "_DelayMatch1DResponse": 10.0,
}

pos_states = {}
for lvl in levels:
    for fam in families_unique:
        nodes_lvl_fam = [
            s for s in range(n_states)
            if state_comp_counts[s] == lvl and state_family[s] == fam
        ]
        nL = len(nodes_lvl_fam)
        if nL == 0: continue
        xs_local = [0.0] if nL == 1 else np.linspace(-4.0, 4.0, nL)
        base = fam_offset[fam]
        for x_local, s in zip(xs_local, nodes_lvl_fam):
            pos_states[s] = (base + x_local, -lvl)

# ---------------------------------------------------------------------
# Manual Layout Adjustments (Optional)
# ---------------------------------------------------------------------
def swap_states_x(task_a, task_b):
    sid_a, sid_b = None, None
    for sid, tasks in enumerate(state_tasks):
        if task_a in tasks: sid_a = sid
        if task_b in tasks: sid_b = sid
    if sid_a is None or sid_b is None: return
    xa, ya = pos_states[sid_a]
    xb, yb = pos_states[sid_b]
    pos_states[sid_a] = (xb, ya)
    pos_states[sid_b] = (xa, yb)

# Try to swap if they exist
swap_states_x("anti", "dlygo")

# ---------------------------------------------------------------------
# 5. Styling
# ---------------------------------------------------------------------

family_palette = {
    "_Reach": "#fde0dd",
    "_DMFamily": "#e0ecf4",
    "_DelayMatch1DResponse": "#e5f5e0",
}
state_colors = [family_palette.get(fam, "#e0e0e0") for fam in state_family]

primitive_display = {
    "integration":      "Rotate",
    "sequence":         "Sequence",
    "anti":             "Anti",
    "has_delay":        "Delay",
    "stim_in_decision": "RT",
    "is_context_mod":   "Context",
    "is_multi_mod":     "Multi-Sensory",
    "is_category_rule": "Category",
    "is_nonmatch_go":   "Non-Match",
    "two_stim_comp":    "Compare 2",
}
primitive_colors = {
    "anti": "#756bb1", "has_delay": "#1d741b", "stim_in_decision": "#a65628",
    "is_context_mod": "#1f78b4", "is_multi_mod": "#dcbeff",
    "is_category_rule": "#6a3d9a", "is_nonmatch_go": "#88ca5e",
    "two_stim_comp": "#e31a1c"
}

state_sizes = [1500 for _ in state_tasks]
state_labels = {sid: "\n".join(tasks) for sid, tasks in enumerate(state_tasks)}

# ---------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------

def plot_state_graph(edges, title):
    G = nx.DiGraph()
    G.add_nodes_from(range(n_states))
    for s_from, s_to, feat in edges:
        G.add_edge(s_from, s_to, feature=feat)

    plt.figure(figsize=(12, 8))

    # Edges
    edge_features = sorted(set(feat for _, _, feat in edges))
    for feat in edge_features:
        sub_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["feature"] == feat]
        if not sub_edges: continue
        nx.draw_networkx_edges(
            G, pos_states, edgelist=sub_edges,
            edge_color=primitive_colors.get(feat, "#bbbbbb"),
            arrowstyle="-|>", arrowsize=15, width=2.0, alpha=0.7,
            connectionstyle="arc3,rad=0.1"
        )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos_states, node_color=state_colors, node_size=state_sizes,
        edgecolors="#666666", linewidths=1.0, alpha=1.0
    )
    nx.draw_networkx_labels(G, pos_states, labels=state_labels, font_size=8, font_weight='bold')

    # Legend
    for feat in edge_features:
        plt.plot([], [], color=primitive_colors.get(feat, "#bbbbbb"), label=primitive_display.get(feat, feat), linewidth=3)
    plt.legend(title="Primitive Added", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("analysis/Taskonomy/Figures/yang19_taskonomy.png")
    print("Saved yang19_taskonomy.png")
    plt.show()

plot_state_graph(edges_states_strict, "Yang19 Task Compositionality (Strict Edges)")

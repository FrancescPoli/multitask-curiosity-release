#!/usr/bin/env python
"""
Family-specific state-level compositionality graphs for Mod-Cog tasks,
using modcog_fundamental_struct2.csv.

Compositional primitives (define states, layers, edges):
  - integration         → Rotate during delay
  - sequence            → Moving target (sequence)
  - anti                → Anti-mapping
  - has_delay           → Delay interval
  - stim_in_decision    → RT (stim in decision)
  - is_context_mod      → Contextual modality (ctxdm*)
  - is_multi_mod        → Multisensory integration (multidm*)
  - is_category_rule    → Category rule (dmc/dnmc)
  - is_nonmatch_go      → Non-match-go (dnms/dnmc)

Complexity features (for "how many processes?"; not used directly for edges):
  - all above
  - plus: has_stim2, has_cohs, w_mod, has_match_rule, has_matchgo

State key = (family, comp_vector):
  → tasks of the same family with identical primitives share a node.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------
# 1. Load CSV and derive extra primitives
# ---------------------------------------------------------------------

df = pd.read_csv("analysis/Taskonomy/Data/modcog_fundamental_struct2.csv")

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

# Complexity features (includes extras)
complexity_cols = [
    "integration",
    "sequence",
    "anti",
    "has_delay",
    "stim_in_decision",
    "has_stim2",
    "has_cohs",
    "w_mod",
    "has_match_rule",
    "has_matchgo",
    "is_context_mod",
    "is_multi_mod",
    "is_category_rule",
    "is_nonmatch_go",
]

# Ensure numeric 0/1 where relevant
df[comp_cols] = df[comp_cols].fillna(0).astype(int)
df[complexity_cols] = df[complexity_cols].fillna(0).astype(int)

# Compositional count (for vertical layering)
df["comp_count"] = df[comp_cols].sum(axis=1)

# Optional complexity score
df["complexity_score"] = df[complexity_cols].sum(axis=1)

print(f"Loaded {len(df)} tasks.")
print("Compositional primitives:", comp_cols)

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
print(f"Found {n_states} distinct compositional states (family + primitives).")

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

state_comp_vectors = np.stack(state_comp_vectors, axis=0)
state_comp_counts = np.array(state_comp_counts, dtype=int)

print("Example state[0] primitives:", dict(zip(comp_cols, state_comp_vectors[0])))

# ---------------------------------------------------------------------
# 3. Build compositional edges between STATES
# ---------------------------------------------------------------------

def build_edges_states_strict():
    """
    State-level edges only if:
      - same family,
      - comp_count_to = comp_count_from + 1,
      - exactly one compositional primitive differs.
    """
    edges = []  # (state_from, state_to, feature_name)

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue

            if state_family[i] != state_family[j]:
                continue

            vi = state_comp_vectors[i]
            vj = state_comp_vectors[j]

            # vj must be strict superset of vi
            if not (np.all(vi <= vj) and np.any(vi < vj)):
                continue

            if state_comp_counts[j] != state_comp_counts[i] + 1:
                continue

            diff_idx = np.where(vi != vj)[0]
            if len(diff_idx) != 1:
                continue

            feat_name = comp_cols[diff_idx[0]]
            edges.append((i, j, feat_name))

    return edges


def build_edges_states_one_diff():
    """
    State-level edges whenever:
      - same family,
      - states differ in exactly one compositional primitive bit,
      - direction from lower comp_count to higher comp_count.
    """
    edges = []  # (state_from, state_to, feature_name)

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue

            if state_family[i] != state_family[j]:
                continue

            vi = state_comp_vectors[i]
            vj = state_comp_vectors[j]

            diff_idx = np.where(vi != vj)[0]
            if len(diff_idx) != 1:
                continue

            feat_name = comp_cols[diff_idx[0]]

            if state_comp_counts[i] < state_comp_counts[j]:
                edges.append((i, j, feat_name))
            elif state_comp_counts[j] < state_comp_counts[i]:
                edges.append((j, i, feat_name))
            # equal comp_count + one bit difference can't happen with 0/1 bits

    # de-duplicate
    edges_unique = list({(i, j, f) for (i, j, f) in edges})
    return edges_unique


edges_states_strict = build_edges_states_strict()
edges_states_one_diff = build_edges_states_one_diff()

print(f"Strict state graph: {len(edges_states_strict)} edges.")
print(f"One-diff state graph: {len(edges_states_one_diff)} edges.")

# ---------------------------------------------------------------------
# 4. Layout: y = -comp_count, x spread within each level and family
# ---------------------------------------------------------------------

levels = sorted(set(state_comp_counts.tolist()))
families_unique = ["_Reach", "_DMFamily", "_DelayMatch1DResponse"]
fam_offset = {
    "_Reach": -10.0,                # left
    "_DMFamily": 0.0,               # centre
    "_DelayMatch1DResponse": 10.0,  # right
}

pos_states = {}
for lvl in levels:
    for fam in families_unique:
        nodes_lvl_fam = [
            s for s in range(n_states)
            if state_comp_counts[s] == lvl and state_family[s] == fam
        ]
        nL = len(nodes_lvl_fam)
        if nL == 0:
            continue
        xs_local = [0.0] if nL == 1 else np.linspace(-4.0, 4.0, nL)
        base = fam_offset[fam]
        for x_local, s in zip(xs_local, nodes_lvl_fam):
            pos_states[s] = (base + x_local, -lvl)

# ---------------------------------------------------------------------
# 4b. Manually swap horizontal positions of 'anti' and 'dlygo' (Reach)
# ---------------------------------------------------------------------

def swap_states_x(task_a: str, task_b: str):
    """Swap x-positions of the states containing task_a and task_b."""
    sid_a = None
    sid_b = None

    for sid, tasks in enumerate(state_tasks):
        if state_family[sid] != "_Reach":
            continue
        if task_a in tasks:
            sid_a = sid
        if task_b in tasks:
            sid_b = sid

    if sid_a is None or sid_b is None:
        return  # one of them not found; do nothing

    xa, ya = pos_states[sid_a]
    xb, yb = pos_states[sid_b]

    # keep the same y (complexity layer), just swap x
    pos_states[sid_a] = (xb, ya)
    pos_states[sid_b] = (xa, yb)

# swap 'anti' and 'dlygo' horizontally
swap_states_x("anti", "dlygo")
swap_states_x("antiseql", "dlyanti")
#swap_states_x("dlyantiintl", "dlygoseql")


# ---------------------------------------------------------------------
# 5. Colours: pastel families for nodes, pastel primitives for edges
# ---------------------------------------------------------------------

family_palette = {
    "_Reach": "#fde0dd",              # light pink
    "_DMFamily": "#e0ecf4",           # light blue
    "_DelayMatch1DResponse": "#e5f5e0",  # light green
}


state_colors = [family_palette.get(fam, "#e0e0e0") for fam in state_family]

# Pastel edge colours and human-readable labels
primitive_display = {
    "integration":      "Rotate during delay",
    "sequence":         "Moving target (sequence)",
    "anti":             "Anti-mapping",
    "has_delay":        "Delay interval",
    "stim_in_decision": "RT (stim in decision)",
    "is_context_mod":   "Contextual modality",
    "is_multi_mod":     "Multisensory integration",
    "is_category_rule": "Category rule",
    "is_nonmatch_go":   "Non-match-go",
}

# Darker, but still somewhat soft, and distinct from node colors
primitive_colors = {
    "integration":      "#de2d26",  # red
    "sequence":         "#ff7f00",  # orange
    "anti":             "#756bb1",  # purple
    "has_delay":        "#1d741b",  # green
    "stim_in_decision": "#a65628",  # brown
    "is_context_mod":   "#1f78b4",  # blue
    "is_multi_mod":     "#dcbeff",  # teal
    "is_category_rule": "#6a3d9a",  # another purple
    "is_nonmatch_go":   "#88ca5e",  # lavander
}


# Node sizes: bigger, to fit multi-line labels
state_sizes = [1200 + 400 * len(tasks) for tasks in state_tasks]

# Labels: all task names in the state, one per line
state_labels = {sid: "\n".join(tasks) for sid, tasks in enumerate(state_tasks)}

# ---------------------------------------------------------------------
# 6. Plot helper
# ---------------------------------------------------------------------

def plot_state_graph(edges, title_suffix):
    G = nx.DiGraph()
    G.add_nodes_from(range(n_states))
    for s_from, s_to, feat in edges:
        G.add_edge(s_from, s_to, feature=feat)

    plt.figure(figsize=(15, 11))

    # Draw edges grouped by feature (with margins for big nodes)
    edge_features = sorted(set(feat for _, _, feat in edges))
    for feat in edge_features:
        sub_edges = [
            (u, v) for (u, v, d) in G.edges(data=True) if d["feature"] == feat
        ]
        if not sub_edges:
            continue
        nx.draw_networkx_edges(
            G,
            pos_states,
            edgelist=sub_edges,
            edge_color=primitive_colors.get(feat, "#bbbbbb"),
            arrowstyle="-|>",
            arrowsize=6,          # smaller heads
            width=2.5,
            alpha=0.8,
            min_source_margin=20, # keep line away from start node
            min_target_margin=25, # keep arrow tip outside big nodes
        )

    # Draw nodes (huge, pastel)
    nx.draw_networkx_nodes(
        G,
        pos_states,
        node_color=state_colors,
        node_size=state_sizes,
        edgecolors="#888888",
        linewidths=0.7,
        alpha=1.0,
    )
    nx.draw_networkx_labels(
        G,
        pos_states,
        labels=state_labels,
        font_size=5,
    )

    # Legend for primitives (edges) with human-readable labels
    for feat in edge_features:
        plt.plot(
            [],
            [],
            color=primitive_colors.get(feat, "#bbbbbb"),
            label=primitive_display.get(feat, feat),
        )
    plt.legend(title="Primitive change", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.title(
        f"State-level compositional graph ({title_suffix})\n"
        "Nodes = (family, primitive pattern), label = all tasks in that state,\n"
        "layers = # active primitives."
    )
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 7. Plot both graphs
# ---------------------------------------------------------------------

plot_state_graph(
    edges_states_strict,
    "STRICT: Δcomp_count = 1, exactly one primitive added",
)

plot_state_graph(
    edges_states_one_diff,
    "ONE-DIFF: any pair differing in exactly one primitive (oriented by comp_count)",
)

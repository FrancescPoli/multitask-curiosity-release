#!/usr/bin/env python
"""
Family-specific state-level compositionality graphs for Poli tasks,
using poli_struct.csv.
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

df = pd.read_csv("analysis/Taskonomy/Data/poli_struct.csv")

# Derive modality-role primitives from w_mod_type
if "w_mod_type" not in df.columns:
    raise ValueError("Expected column 'w_mod_type' not found in poli_struct.csv")

df["is_context_mod"] = df["w_mod_type"].isin(["context_mod1", "context_mod2"]).astype(int)
df["is_multi_mod"] = (df["w_mod_type"] == "multi_sum").astype(int)

# Derive match-rule primitives (DelayMatch family)
df["is_category_rule"] = (df["match_rule_type"] == "category").astype(int)
df["is_nonmatch_go"] = (df["matchgo_type"] == "non_match").astype(int)

# Two-stimulus / comparison primitive:
df["two_stim_comp"] = (
    (df["has_cohs"] == 1) |
    ((df["has_stim2"] == 1) & (df["has_match_rule"] == 1))
).astype(int)

# Compositional primitives (Restricted to Poli modifiers for vertical layering)
comp_cols = [
    "anti",
    "has_delay",
    "stim_in_decision", # rt
    "is_context_mod",   # ctx
    "is_multi_mod",     # multi
    "is_category_rule", # cat
    "is_nonmatch_go",   # nms
    # "two_stim_comp",  # Excluded per user request
]

# Ensure numeric 0/1 where relevant
df[comp_cols] = df[comp_cols].fillna(0).astype(int)

# Compositional count (for vertical layering)
df["comp_count"] = df[comp_cols].sum(axis=1)

print(f"Loaded {len(df)} tasks.")
print("Compositional primitives (active):", comp_cols)

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
    "_Reach": -0.6,                # Left
    "_DMFamily": 0.0,               # Center
    "_DelayMatch1DResponse": 0.6,   # Right
}
# Adjusted offsets to keep closer? Or keep wide? Original was -10, 0, 10.
# I'll stick to original unless plotting issue.
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
        if nL == 0:
            continue
        xs_local = [0.0] if nL == 1 else np.linspace(-4.0, 4.0, nL)
        base = fam_offset[fam]
        for x_local, s in zip(xs_local, nodes_lvl_fam):
            pos_states[s] = (base + x_local, -lvl)

# ---------------------------------------------------------------------
# 4b. Manually swap horizontal positions (Optional - Commented out for Poli)
# ---------------------------------------------------------------------

def swap_states_x(task_a: str, task_b: str):
    sid_a = None
    sid_b = None
    for sid, tasks in enumerate(state_tasks):
        if state_family[sid] != "_Reach": continue
        if task_a in tasks: sid_a = sid
        if task_b in tasks: sid_b = sid
    if sid_a is None or sid_b is None: return

    xa, ya = pos_states[sid_a]
    xb, yb = pos_states[sid_b]
    pos_states[sid_a] = (xb, ya)
    pos_states[sid_b] = (xa, yb)

# swap_states_x("poli.antigo", "poli.dlygo") # Example if needed

# ---------------------------------------------------------------------
# 5. Colours
# ---------------------------------------------------------------------

family_palette = {
    "_Reach": "#fde0dd",              # light pink
    "_DMFamily": "#e0ecf4",           # light blue
    "_DelayMatch1DResponse": "#e5f5e0",  # light green
}

state_colors = [family_palette.get(fam, "#e0e0e0") for fam in state_family]

primitive_display = {
    "anti":             "Anti-Mapping (A)",
    "has_delay":        "Delay Interval (I)",
    "is_context_mod":   "Distractor (D)",
    "stim_in_decision": "RT (Stimulus after fixation) (R)",
    "is_multi_mod":     "Multisensory Integration (M)",
    "is_category_rule": "Categorical Mapping (C)",
    "is_nonmatch_go":   "Non-match (N)",
}

legend_order = [
    "anti", "has_delay", "is_context_mod", "stim_in_decision", 
    "is_multi_mod", "is_category_rule", "is_nonmatch_go"
]

primitive_colors = {
    "integration":      "#de2d26",
    "sequence":         "#ff7f00",
    "anti":             "#de2d26", # A (Red)
    "has_delay":        "#1d741b", # I (Green)
    "is_context_mod":   "#1f78b4", # D (Blue)
    "stim_in_decision": "#a65628", # R (Brown)
    "is_multi_mod":     "#dcbeff", # M (Teal/Purple)
    "is_category_rule": "#ff7f00", # C (Orange)
    "is_nonmatch_go":   "#88ca5e", # N (Light Green)
}

# Node sizes: bigger
state_sizes = [1500 + 300 * len(tasks) for tasks in state_tasks]

def shorten_name(name):
    """Shorten task names: antictxgo -> ADgo"""
    name = name.replace("poli.", "")
    # Replacements (Careful with order)
    # antictxgo: anti -> A, ctx -> D
    # dly: I
    
    # We must replace modifers but keep base.
    # Modifiers: anti, ctx, dly, rt, multi, cat, nms
    # Map:
    subs = [
        ("anti", "A"),
        ("ctx", "D"),
        ("dly", "I"),
        ("rt", "R"),
        ("multi", "M"),
        ("cat", "C"),
        ("nms", "Nms"), # Keep ms suffix for non-match
    ]
    for old, new in subs:
        name = name.replace(old, new)
    return name

# Labels: all task names in the state, one per line, shortened
state_labels = {
    sid: "\n".join(shorten_name(t) for t in tasks) 
    for sid, tasks in enumerate(state_tasks)
}

# ---------------------------------------------------------------------
# 6. Plot helper
# ---------------------------------------------------------------------

def plot_state_graph(edges, title_suffix, filename):
    G = nx.DiGraph()
    G.add_nodes_from(range(n_states))
    for s_from, s_to, feat in edges:
        G.add_edge(s_from, s_to, feature=feat)

    plt.figure(figsize=(18, 12)) # Larger
    plt.axis('off') # Remove axis and black box

    edge_features = sorted(set(feat for _, _, feat in edges))
    for feat in edge_features:
        sub_edges = [
            (u, v) for (u, v, d) in G.edges(data=True) if d["feature"] == feat
        ]
        if not sub_edges: continue
        nx.draw_networkx_edges(
            G, pos_states, edgelist=sub_edges,
            edge_color=primitive_colors.get(feat, "#bbbbbb"),
            arrowstyle="-|>", arrowsize=10, width=2.5, alpha=0.8,
            min_source_margin=20, min_target_margin=25,
        )

    nx.draw_networkx_nodes(
        G, pos_states, node_color=state_colors, node_size=state_sizes,
        edgecolors="#888888", linewidths=0.7, alpha=1.0,
    )
    nx.draw_networkx_labels(G, pos_states, labels=state_labels, font_size=8) # removed font_weight='bold'

    # Legend (Top Right, Inside, Custom Order)
    for feat in legend_order:
        if feat in edge_features: # Only plot relevant ones
             plt.plot([], [], color=primitive_colors.get(feat, "#bbbbbb"),
                      label=primitive_display.get(feat, feat))
    plt.legend(title="Primitive", loc="upper right")

    # plt.title(f"Poli Taskonomy ({title_suffix})") # Removed title
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1) # Tight layout to clip whitespace
    # plt.show() # Make sure to save instead of show if headless

# ---------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------

plot_state_graph(edges_states_strict, "STRICT", "analysis/Taskonomy/Figures/poli_taskonomy_strict.png")
# plot_state_graph(edges_states_one_diff, "ONE-DIFF", "poli_taskonomy_onediff.png")
print("Saved poli_taskonomy_strict.png")

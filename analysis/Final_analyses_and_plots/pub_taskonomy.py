"""
Publication-quality Poli taskonomy (strict compositional graph).

Produces TWO files:
  pub_taskonomy_strict.pdf/.png   – all 48 tasks (full graph)
  pub_taskonomy_32.pdf/.png       – 20 training + 12 held-out tasks only,
                                    with nodes colour-coded by split

Key changes vs plot_poli_taskonomy.py:
  - Wider figure, larger fonts.
  - Family names: Reach / Decision-Making / Match-to-Sample.
  - Match-to-Sample shifted up so its top row aligns with the other families.
  - Saves vector PDF + 300 dpi PNG.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Task lists ─────────────────────────────────────────────────────────────────
TRAIN_TASKS = {
    'poli.go', 'poli.rtgo', 'poli.dlygo', 'poli.antigo', 'poli.ctxgo',
    'poli.dlyantigo', 'poli.dlyctxgo', 'poli.antictxgo',
    'poli.dlyantictxgo', 'poli.rtantictxgo',
    'poli.dm1', 'poli.dm2', 'poli.dlydm1', 'poli.multidm',
    'poli.antidm1', 'poli.antidlydm1',
    'poli.dlyms', 'poli.dlynms', 'poli.catdlyms', 'poli.catdlynms',
}
HELD_TASKS = {
    'poli.ctxdm1', 'poli.ctxdm2',
    'poli.ctxdlydm1', 'poli.ctxdlydm2',
    'poli.antictxdm1', 'poli.antictxdm2',
    'poli.antictxdlydm1', 'poli.antictxdlydm2',
    'poli.antidlyms', 'poli.antidlynms',
    'poli.antictxdlyms', 'poli.antictxcatdlyms',
}
KEEP_32 = TRAIN_TASKS | HELD_TASKS

# ── Load & derive ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
df = pd.read_csv(os.path.join(ROOT, "analysis/Taskonomy/Data/poli_struct.csv"))

df["is_context_mod"]   = df["w_mod_type"].isin(["context_mod1", "context_mod2"]).astype(int)
df["is_multi_mod"]     = (df["w_mod_type"] == "multi_sum").astype(int)
df["is_category_rule"] = (df["match_rule_type"] == "category").astype(int)
df["is_nonmatch_go"]   = (df["matchgo_type"] == "non_match").astype(int)

COMP_COLS = ["anti", "has_delay", "stim_in_decision",
             "is_context_mod", "is_multi_mod", "is_category_rule", "is_nonmatch_go"]
df[COMP_COLS] = df[COMP_COLS].fillna(0).astype(int)
df["comp_count"] = df[COMP_COLS].sum(axis=1)

# ── Build states ───────────────────────────────────────────────────────────────
state_groups: dict = defaultdict(list)
for idx, row in df.iterrows():
    state_groups[(row["family"], tuple(int(row[c]) for c in COMP_COLS))].append(idx)

state_keys     = list(state_groups.keys())
n_states       = len(state_keys)
state_vecs     = np.stack([np.array(v, int) for _, v in state_keys])
state_counts   = state_vecs.sum(axis=1)
state_families = [f for f, _ in state_keys]
state_tasks    = [[df.loc[i, "name"] for i in state_groups[k]] for k in state_keys]

# ── Strict edges ───────────────────────────────────────────────────────────────
def build_strict_edges(allowed_states=None):
    edges = []
    for i in range(n_states):
        for j in range(n_states):
            if i == j or state_families[i] != state_families[j]:
                continue
            if allowed_states is not None and (i not in allowed_states or j not in allowed_states):
                continue
            vi, vj = state_vecs[i], state_vecs[j]
            if not (np.all(vi <= vj) and state_counts[j] == state_counts[i] + 1):
                continue
            diff = np.where(vi != vj)[0]
            if len(diff) == 1:
                edges.append((i, j, COMP_COLS[diff[0]]))
    return edges

# ── Colours ────────────────────────────────────────────────────────────────────
FAMILIES = ["_Reach", "_DMFamily", "_DelayMatch1DResponse"]

FAM_NAME = {
    "_Reach":                   "Reach",
    "_DMFamily":                "Decision-Making",
    "_DelayMatch1DResponse":    "Match-to-Sample",
}
FAM_FACE = {
    "_Reach":                   "#fde0dd",
    "_DMFamily":                "#ddeeff",
    "_DelayMatch1DResponse":    "#e5f5e0",
}
# held-out node faces (slightly darker shade of family colour)
FAM_FACE_HELD = {
    "_Reach":                   "#f4a78a",
    "_DMFamily":                "#7bafd4",
    "_DelayMatch1DResponse":    "#88c97a",
}

PRIM_COLOR = {
    "anti":             "#d62728",
    "has_delay":        "#2ca02c",
    "stim_in_decision": "#8c564b",
    "is_context_mod":   "#1f77b4",
    "is_multi_mod":     "#9467bd",
    "is_category_rule": "#ff7f0e",
    "is_nonmatch_go":   "#17becf",
}
PRIM_LABEL = {
    "anti":             "A: Anti-mapping",
    "has_delay":        "I: Delay interval",
    "stim_in_decision": "R: Reaction time",
    "is_context_mod":   "D: Distractor",
    "is_multi_mod":     "M: Multisensory",
    "is_category_rule": "C: Categorical rule",
    "is_nonmatch_go":   "N: Non-match",
}
LEGEND_ORDER = ["anti", "has_delay", "stim_in_decision",
                "is_context_mod", "is_multi_mod", "is_category_rule", "is_nonmatch_go"]

# ── Label shortener ────────────────────────────────────────────────────────────
_REPLACEMENTS = [
    ("antictx", "AD"), ("anticat", "AC"), ("multidly", "MI"),
    ("anti", "A"), ("ctx", "D"), ("multi", "M"),
    ("dly",  "I"), ("cat",  "C"), ("rt",   "R"),
]
def shorten(name: str) -> str:
    name = name.replace("poli.", "")
    for old, new in _REPLACEMENTS:
        name = name.replace(old, new)
    return name

# ── Layout builder ─────────────────────────────────────────────────────────────
FAM_X = {"_Reach": -14.0, "_DMFamily": 0.0, "_DelayMatch1DResponse": 14.0}
# Match-to-Sample starts at comp_count=1; shift up so top row = y=0 like others
FAM_Y_OFFSET = {"_DelayMatch1DResponse": 1.8}

def build_pos(allowed_states=None):
    pos: dict = {}
    for lvl in sorted(set(state_counts.tolist())):
        for fam in FAMILIES:
            nodes = [s for s in range(n_states)
                     if state_counts[s] == lvl and state_families[s] == fam
                     and (allowed_states is None or s in allowed_states)]
            if not nodes:
                continue
            # Wider spread so nodes at the same level don't overlap
            spread = max(5.5, 1.3 * len(nodes))
            xs = [0.0] if len(nodes) == 1 else np.linspace(-spread, spread, len(nodes))
            y_off = FAM_Y_OFFSET.get(fam, 0.0)
            for x, s in zip(xs, nodes):
                pos[s] = (FAM_X[fam] + x, -lvl * 1.8 + y_off)

    # Surgical: place ADCIms directly under ADIms (align x only)
    def _find_state(task_name):
        for s, tasks in enumerate(state_tasks):
            if task_name in tasks:
                return s
        return None

    s_adims  = _find_state("poli.antictxdlyms")       # ADIms
    s_adcims = _find_state("poli.antictxcatdlyms")    # ADCIms
    if (s_adims is not None and s_adcims is not None
            and s_adims in pos and s_adcims in pos):
        pos[s_adcims] = (pos[s_adims][0], pos[s_adcims][1])

    return pos

# ── Figure size ────────────────────────────────────────────────────────────────
FIG_H    = 4.5    # base height in inches
WH_RATIO = 2    # width / height  — increase to widen, decrease to narrow
FIG_W    = FIG_H * WH_RATIO

# ── Plot function ──────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

def plot_taskonomy(allowed_states=None, node_split=None, outname="pub_taskonomy"):
    """
    allowed_states : set of state indices to include (None = all)
    node_split     : dict {state_idx: 'train'|'held'} for colour coding (None = family colour only)
    outname        : output filename stem (no extension)
    """
    states = list(range(n_states)) if allowed_states is None else sorted(allowed_states)
    pos    = build_pos(allowed_states=set(states))
    edges  = build_strict_edges(allowed_states=set(states))

    # Node colours
    def node_color(s):
        if node_split is not None:
            split = node_split.get(s, "train")
            if split == "held":
                return FAM_FACE_HELD.get(state_families[s], "#cccccc")
        return FAM_FACE.get(state_families[s], "#e0e0e0")

    n_colors = [node_color(s) for s in states]
    n_sizes  = [900 + 200 * len(state_tasks[s]) for s in states]
    n_labels = {s: "\n".join(shorten(t) for t in state_tasks[s]) for s in states}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.axis("off")

    G = nx.DiGraph()
    G.add_nodes_from(states)

    edge_feats = sorted({f for _, _, f in edges})
    for feat in edge_feats:
        sub = [(u, v) for u, v, f in edges if f == feat]
        nx.draw_networkx_edges(
            G, pos, edgelist=sub, ax=ax,
            edge_color=PRIM_COLOR[feat],
            arrowstyle="-|>", arrowsize=12, width=2.2, alpha=0.85,
            min_source_margin=18, min_target_margin=22,
        )

    nx.draw_networkx_nodes(G, pos, nodelist=states, ax=ax,
                           node_color=n_colors, node_size=n_sizes,
                           edgecolors="#666666", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels=n_labels, ax=ax,
                            font_size=7.5, font_color="#111111")

    # Family header labels — place at the top of each family's column
    for fam in FAMILIES:
        fam_nodes = [s for s in states if state_families[s] == fam]
        if not fam_nodes:
            continue
        top_y = max(pos[s][1] for s in fam_nodes)
        ax.text(FAM_X[fam], top_y + 0.6, FAM_NAME[fam],
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#222222")

    # Legend for primitives
    for feat in LEGEND_ORDER:
        if feat in edge_feats:
            ax.plot([], [], color=PRIM_COLOR[feat], lw=2.5, label=PRIM_LABEL[feat])

    legend_handles = [ax.get_legend_handles_labels()[0][i]
                      for i in range(len(ax.get_legend_handles_labels()[0]))]
    legend_labels  = ax.get_legend_handles_labels()[1]

    # For split version: add training/held-out patches
    if node_split is not None:
        legend_handles += [
            mpatches.Patch(facecolor="#e0e0e0", edgecolor="#666", label="Training task"),
            mpatches.Patch(facecolor="#aaaaaa", edgecolor="#666", label="Held-out task"),
        ]
        legend_labels += ["Training task", "Held-out task"]

    ax.legend(legend_handles, legend_labels,
              title="Compositional primitive", title_fontsize=9,
              loc="center left", bbox_to_anchor=(1.02, 0.5),
              ncol=1, fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="#cccccc")

    plt.tight_layout()
    OUT = os.path.join(os.path.dirname(__file__), "Figures", outname)
    plt.savefig(OUT + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT}.pdf / .png")
    plt.close()

# ── Version 1: full graph ──────────────────────────────────────────────────────
plot_taskonomy(outname="pub_taskonomy_strict")

# ── Version 2: 20 training + 12 held-out only ─────────────────────────────────
# Determine which states to include and their split label
allowed32  = set()
split_map  = {}

for s in range(n_states):
    tasks = state_tasks[s]
    in_train = any(t in TRAIN_TASKS for t in tasks)
    in_held  = any(t in HELD_TASKS  for t in tasks)
    if in_train or in_held:
        allowed32.add(s)
        # If node has any held-out task, mark as held; otherwise train
        split_map[s] = "held" if (in_held and not in_train) else "train"

plot_taskonomy(allowed_states=allowed32, node_split=split_map, outname="pub_taskonomy_32")

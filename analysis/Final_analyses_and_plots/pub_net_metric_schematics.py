"""
Publication-quality schematics of the three network metrics used in Figure 3.

One shared 12-node network is drawn three times with different highlighting:
  - Modularity       : nodes coloured by module
  - Rich Club        : hub triangle highlighted
  - Global Efficiency: one shortest peripheral-to-peripheral path highlighted
                       (routed through the rich club)

The three hubs are simultaneously the inter-module bridges AND the rich club,
so highlighting different subsets of the same graph tells three coherent
stories about the same architecture.

Outputs (PNG, 400 dpi) in analysis/Final_analyses_and_plots/Figures/ :
  pub_net_metric_schematics.png        – triptych
  pub_net_metric_modularity.png
  pub_net_metric_rich_club.png
  pub_net_metric_efficiency.png
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Tweakables ───────────────────────────────────────────────────────────────
PANEL_W = 2.0      # inches per panel
PANEL_H = 2.0
DPI     = 400

NODE_R      = 0.075     # peripheral / default node radius (axes units)
HUB_R       = 0.115     # hub radius (rich-club panel only; elsewhere = NODE_R)
NODE_LW     = 0.9       # node outline
EDGE_LW     = 1.0       # default edge width
HL_LW       = 2.6       # highlighted edge width

# Palette
NEUTRAL_EDGE    = "#8A8A8A"
DIM_EDGE        = "#8A8A8A"
DARK_OUTLINE    = "#1F1F1F"
NEUTRAL_FILL    = "#FFFFFF"

CLUSTER_COLS    = ["#4C78A8", "#F2A93B", "#54A24B"]   # blue / gold / green
HIGHLIGHT       = "#E26714"                           # orange (matches rest of figure)
HUB_FILL        = "#2B3035"
PERI_FILL       = "#E8E8E8"

TITLE_SIZE  = 10
SHOW_TITLES = True

OUT_DIR = Path("analysis/Final_analyses_and_plots/Figures")


# ── Shared geometry: ONE graph, used by all three panels ─────────────────────
# 3 modules × 4 nodes. Node name convention: f"{M}{k}" where k=0 is the hub.
CLUSTERS: dict[str, list[str]] = {
    "A": ["A0", "A1", "A2", "A3"],
    "B": ["B0", "B1", "B2", "B3"],
    "C": ["C0", "C1", "C2", "C3"],
}
HUBS: list[str] = ["A0", "B0", "C0"]

# Node positions: three modules laid out around the centre. Hub sits at the
# inner corner (closest to origin); peripherals fan outward.
def _cluster_positions(centre: tuple[float, float],
                       radial_dir: tuple[float, float],
                       *, hub_offset: float = 0.28,
                       peri_radius: float = 0.42) -> dict[str, tuple[float, float]]:
    cx, cy = centre
    rx, ry = radial_dir
    # hub: shifted inward along -radial_dir
    hub = (cx - rx * hub_offset, cy - ry * hub_offset)
    # peripherals: three points on a small arc around the module centre,
    # on the outward side (so module visually "fans out")
    base_ang = np.arctan2(ry, rx)
    angs = base_ang + np.deg2rad([-40, 0, 40])
    peris = [(cx + peri_radius * np.cos(a), cy + peri_radius * np.sin(a)) for a in angs]
    return {"0": hub, "1": peris[0], "2": peris[1], "3": peris[2]}


POS: dict[str, tuple[float, float]] = {}
# Three modules at angles 90°, 210°, 330° from origin, centre radius = 0.55
for mod, ang_deg in zip(["A", "B", "C"], [90, 210, 330]):
    a = np.deg2rad(ang_deg)
    cx, cy = 0.55 * np.cos(a), 0.55 * np.sin(a)
    for k, p in _cluster_positions((cx, cy), (np.cos(a), np.sin(a))).items():
        POS[f"{mod}{k}"] = p


# Edges: intra-module clique (all 6 edges among 4 nodes) + hub triangle.
EDGES: list[tuple[str, str]] = []
for mod, nodes in CLUSTERS.items():
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            EDGES.append((nodes[i], nodes[j]))
# hub triangle
EDGES += [("A0", "B0"), ("B0", "C0"), ("A0", "C0")]


def node_cluster(name: str) -> str:
    return name[0]


def is_hub(name: str) -> bool:
    return name in HUBS


def is_intra(u: str, v: str) -> bool:
    return node_cluster(u) == node_cluster(v)


# Shortest path for the efficiency panel. Far peripheral in A to far peripheral
# in C, routed through the rich club. Chosen so the path reads left-to-right.
EFF_PATH_NODES = ["A1", "A0", "C0", "C3"]
EFF_PATH_EDGES = set()
for u, v in zip(EFF_PATH_NODES[:-1], EFF_PATH_NODES[1:]):
    EFF_PATH_EDGES.add(frozenset((u, v)))


# ── Drawing primitives ───────────────────────────────────────────────────────
def _edge_kwargs(col, lw, alpha=1.0, z=1):
    return dict(color=col, lw=lw, solid_capstyle="round", alpha=alpha, zorder=z)


def draw_base(ax,
              *,
              edge_style: Callable[[str, str], dict],
              node_style: Callable[[str], dict]) -> None:
    # Edges first so nodes cover them.
    # Two-pass: low-zorder edges first, highlighted edges on top.
    deferred = []
    for u, v in EDGES:
        s = edge_style(u, v)
        if s.get("z", 1) >= 2:
            deferred.append((u, v, s))
            continue
        p, q = POS[u], POS[v]
        ax.plot([p[0], q[0]], [p[1], q[1]], **_edge_kwargs(**s))
    for u, v, s in deferred:
        p, q = POS[u], POS[v]
        ax.plot([p[0], q[0]], [p[1], q[1]], **_edge_kwargs(**s))

    for name, p in POS.items():
        s = node_style(name)
        ax.add_patch(mpatches.Circle(
            p,
            radius=s.get("r", NODE_R),
            facecolor=s["fc"],
            edgecolor=s["ec"],
            linewidth=s.get("lw", NODE_LW),
            zorder=s.get("z", 3),
        ))


def style_axis(ax, title=None):
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal"); ax.axis("off")
    if title and SHOW_TITLES:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=2)


# ── Panel A: Modularity ──────────────────────────────────────────────────────
def panel_modularity(ax):
    cluster_colour = {m: CLUSTER_COLS[i] for i, m in enumerate(CLUSTERS)}

    def edge_style(u, v):
        if is_intra(u, v):
            return dict(col=cluster_colour[node_cluster(u)], lw=EDGE_LW, alpha=0.9, z=1)
        return dict(col=DIM_EDGE, lw=EDGE_LW * 0.85, alpha=1.0, z=1)

    def node_style(name):
        c = cluster_colour[node_cluster(name)]
        return dict(fc=c, ec=DARK_OUTLINE, r=NODE_R, lw=NODE_LW, z=3)

    draw_base(ax, edge_style=edge_style, node_style=node_style)
    style_axis(ax, "Modularity")


# ── Panel B: Rich Club ───────────────────────────────────────────────────────
def panel_rich_club(ax):
    def edge_style(u, v):
        if is_hub(u) and is_hub(v):
            return dict(col=HIGHLIGHT, lw=HL_LW, alpha=1.0, z=2)
        return dict(col=DIM_EDGE, lw=EDGE_LW * 0.85, alpha=1.0, z=1)

    def node_style(name):
        if is_hub(name):
            return dict(fc=HUB_FILL, ec=DARK_OUTLINE, r=HUB_R, lw=NODE_LW, z=4)
        return dict(fc=PERI_FILL, ec="#8A8A8A", r=NODE_R * 0.85, lw=NODE_LW, z=3)

    draw_base(ax, edge_style=edge_style, node_style=node_style)
    style_axis(ax, "Rich Club")


# ── Panel C: Global Efficiency ───────────────────────────────────────────────
EFF_NODE_FILL = "#D9D9D9"   # gray fill for non-endpoint nodes


def panel_efficiency(ax):
    endpoints = {EFF_PATH_NODES[0], EFF_PATH_NODES[-1]}

    def edge_style(u, v):
        if frozenset((u, v)) in EFF_PATH_EDGES:
            return dict(col=HIGHLIGHT, lw=HL_LW, alpha=1.0, z=2)
        return dict(col=DIM_EDGE, lw=EDGE_LW * 0.85, alpha=1.0, z=1)

    def node_style(name):
        if name in endpoints:
            return dict(fc=HIGHLIGHT, ec=DARK_OUTLINE, r=NODE_R, lw=NODE_LW, z=4)
        return dict(fc=EFF_NODE_FILL, ec=DARK_OUTLINE, r=NODE_R, lw=NODE_LW, z=3)

    draw_base(ax, edge_style=edge_style, node_style=node_style)
    style_axis(ax, "Global Efficiency")


# ── Render ───────────────────────────────────────────────────────────────────
PANELS = [
    ("modularity", panel_modularity),
    ("rich_club",  panel_rich_club),
    ("efficiency", panel_efficiency),
]


def render_triptych():
    fig, axes = plt.subplots(3, 1, figsize=(PANEL_W, PANEL_H * 3))
    for ax, (_, fn) in zip(axes, PANELS):
        fn(ax)
    fig.subplots_adjust(hspace=0.08, left=0.02, right=0.98, top=0.98, bottom=0.01)
    out = OUT_DIR / "pub_net_metric_schematics.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}")


def render_individual():
    for name, fn in PANELS:
        fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
        fn(ax)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        out = OUT_DIR / f"pub_net_metric_{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out.name}")


def main():
    render_triptych()
    render_individual()


if __name__ == "__main__":
    main()

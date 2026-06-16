# Multitask Curiosity

Code and shareable data for a study linking a **curiosity-driven training curriculum** to brain-like organization in multi-task recurrent neural networks (RNNs).

RNNs are trained on a 20-task cognitive battery in which the order of practice is decided online by a **Marginal Value Theorem (MVT) controller** - the same foraging rule we fit to infants' looking behavior. Trained networks are then compared to humans along three axes:

- **Brain network structure**: RNN recurrent connectivity vs. HCP-YA structural connectomes (Schaefer-100 graph topology: modularity, efficiency, rich club).
- **Synaptic density**: RNN weight-density trajectories vs. the human lifespan synaptic-density curve.
- **Cognitive functioning**: compositional generalization (frozen-weight probes on held-out task recombinations).

The full Methods are in [`methods/`](methods/) (Quarto). Setup and the end-to-end run order are in [`startup_demo.md`](startup_demo.md). Data availability and what is/ isn't redistributable are in [`DATA.md`](DATA.md).

## Repository layout

```         
run.py, paths.py                 training entry point + logs-path resolution
experiments/                     parameter-sweep launchers (SLURM / local)
curiosity/                       model, tasks, and the MVT curriculum controller
analysis/
  Network_analysis/              graph topology (RNN + human); shared topology_metrics.py
  Synaptic_analysis/             weight-evolution extraction + synaptic-space fit
  Cognitive_analysis/            infant MVT fitting + model MVT recovery (+ Goldilocks/)
  compositional_analysis/        cohort building + compositional-generalization analysis
  Taskonomy/                     task-structure / taskonomy graphs
  Final_analyses_and_plots/      consolidated publication scripts + Figures/
  Performance_analysis/, Regularization_scope/, Forest_analysis/   supplementary analyses
  aggregate_all_metrics.py       merges all axes -> grand_unified_metrics_v2.csv
methods/                         manuscript Methods (Quarto -> docx)
```

## Reproducing the main analyses

The reported figures and statistics come from three consolidated scripts, run from the project root after metrics are aggregated (`R 4.2`):

``` bash
RS="C:/Program Files/R/R-4.2.2/bin/Rscript.exe"
& $RS analysis/Final_analyses_and_plots/pub_topology_consolidated.R     # network structure
& $RS analysis/Final_analyses_and_plots/pub_synaptic_consolidated.R     # synaptic density
& $RS analysis/compositional_analysis/pub_compositional_generalisation.R # compositional gen.
```

Each prints its statistics and writes PNGs to `analysis/Final_analyses_and_plots/Figures/`. "Most human-like" is defined everywhere by a single Mahalanobis fingerprint over the three topological metrics (`METRIC_COLS`, kept in sync across the consolidated scripts and the cohort builder).

Training, cluster sweeps, the per-run extraction steps, and the compositional probing are documented step-by-step in [`startup_demo.md`](startup_demo.md).

## Data

This repository ships the **shareable** data needed to reproduce everything downstream of the restricted inputs:

- `analysis/Cognitive_analysis/Data/Infant_Data/infant_foraging_release.csv` — anonymized infant foraging trials (no identifiers; integer `subj_id` only).
- Model-derived metric tables and the derived human topology table.

It **excludes** raw infant data / demographics / IDs and the HCP-derived connectomes, which cannot be redistributed. See [`DATA.md`](DATA.md) for the exact exclusions and how to obtain the restricted inputs (infant data: contact the authors under ethics/consent; HCP data: https://www.humanconnectome.org/).

## Environments

- Python: conda env `ngym39` (PyTorch + neurogym + analysis deps).
- R: 4.2 with `dplyr, tidyr, ggplot2, emmeans, car, mgcv, survival, plot3D` for the publication scripts.

Install details are in [`startup_demo.md`](startup_demo.md) (Part 3).
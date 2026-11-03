# Multitask Curiosity

Code and shareable data for a study linking a **curiosity-driven training curriculum** to brain-like organization in multi-task recurrent neural networks (RNNs).

RNNs are trained on a 20-task cognitive battery in which the order of practice is decided online by a **Marginal Value Theorem (MVT) controller** - the same foraging rule we fit to infants' looking behavior. Trained networks are then compared to humans along three axes:

- **Brain network structure**: RNN recurrent connectivity vs. HCP-YA structural connectomes (Schaefer-100 graph topology: modularity, efficiency, rich club).
- **Synaptic density**: RNN weight-density trajectories vs. the human lifespan synaptic-density curve.
- **Cognitive functioning**: compositional generalization (frozen-weight probes on held-out task recombinations).

The full Methods are in [`methods/`](methods/) (Quarto). The end-to-end pipeline (cluster sweeps, extraction, analysis) is documented step by step in [`startup_demo.md`](startup_demo.md). Data availability and what is/isn't redistributable are in [`DATA.md`](DATA.md).

## Repository layout

```
run.py, paths.py                 training entry point + logs-path resolution
experiments/                     parameter-sweep launchers (SLURM / local)
curiosity/                       model, tasks, and the MVT curriculum controller
demo/                            demo script (see "Demo")
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
Mod_Cog/                         Mod-Cog task definitions (Khona et al., CC0)
```

## 1. System requirements

**Operating systems tested**
- Windows 11 Enterprise (build 26200): demo, analyses, and figures.
- Linux, Ubuntu 24.04 LTS (SLURM cluster): full parameter sweep (training of all networks) and per-run extraction.

**Software**
- Python 3.9 (tested 3.9.24). Packages and tested versions are pinned in [`requirements.txt`](requirements.txt): numpy 1.23.5, pandas 2.3.3, scipy 1.13.1, torch 2.8.0, matplotlib 3.9.4, seaborn 0.13.2, gym 0.25.2, NeuroGym (commit `b060cc2`), scikit-learn 1.6.1, tqdm 4.68.2, networkx 3.2.1, h5py 3.14.0, bctpy 0.6.1, leidenalg 0.12.0, igraph 1.0.0, netneurotools 0.3.0.
- R 4.2 (tested 4.2.2) with: dplyr 1.1.4, tidyr 1.3.1, readr 2.1.5, ggplot2 3.5.1, scales 1.3.0, emmeans 1.10.1, car 3.1-2, MASS 7.3-58.1, mgcv 1.9-1, survival 3.4-0, plot3D 1.4.2. Optional (supplementary/infant scripts): lme4 1.1-35.3, jtools 2.2.2, parameters 0.24.2, modelbased 0.10.0, see 0.11.0.

**Hardware**
- No non-standard hardware is required. The demo runs on a standard desktop CPU (no GPU).
- The full study trained 3,136 networks for 600,000 steps each; each run used 1 CPU core and 8 GB RAM on a SLURM cluster (no GPU).

## 2. Installation guide

On Windows, some file names in the repository are long: clone into a short path (e.g. `C:\code\`) or enable long paths first with `git config --global core.longpaths true`.

```bash
git clone https://github.com/FrancescPoli/multitask-curiosity-release.git
cd multitask-curiosity-release

# Python environment
conda create -n ngym39 python=3.9 -y
conda activate ngym39
pip install -r requirements.txt

# R packages (from an R 4.2 session)
Rscript -e 'install.packages(c("dplyr","tidyr","readr","ggplot2","scales","emmeans","car","MASS","mgcv","survival","plot3D"), repos="https://cloud.r-project.org")'
```

NeuroGym is installed in editable mode from a pinned commit (into `./src/neurogym`); a regular install omits part of the package.

Typical install time on a normal desktop: about 10 minutes (mostly downloading PyTorch); R packages a few additional minutes if not already installed.

## 3. Demo

The demo (i) trains one RNN with the curiosity-driven MVT curriculum, using the same settings as the study but 500 instead of 600,000 training steps; (ii) analyses its connectome; and (iii) reproduces the main statistics and figures of the paper from the shipped metric tables of all 3,136 trained networks.

**Run** (from the repository root, with the `ngym39` environment active):

```bash
bash demo/run_demo.sh
```

On Windows, run it from Git Bash. If `python` or `Rscript` are not on your PATH, pass them explicitly:
`PYTHON=/path/to/python RSCRIPT=/path/to/Rscript bash demo/run_demo.sh`.

**Expected output**

| Step | Output | What to check |
|---|---|---|
| 1. Train | `demo/output/sweep/run_demo/`: checkpoints `state_step000000.pt` … `state_step000500.pt`, `model_last.pt`, `run_curriculum.csv` (the MVT stay/leave log) | Console prints loss/accuracy every 100 steps; accuracy is still low after 500 steps (the full study trains for 600,000) |
| 2. Analyse | `demo/output/population_weights_demo.csv` (weight-density trajectory), `demo/output/rnn_topology_demo.csv` (topology of the final network), `weights_evolution.npy` (6 × 100 × 100) | One row of metrics at density 0.135: modularity, global efficiency, rich club, small-worldness, participation |
| 3. Reproduce | Printed statistics (also saved to `demo/output/pub_*.log`) and PNGs in `analysis/Final_analyses_and_plots/Figures/` (identical to the shipped figures) | Network structure: `Overall: F(11,2368) = 1200.0, R2 = 0.8479`; best cell `curiosity / reg=0.0005 / Distance (s = 0.799)`. Synaptic density: `fitted-age chi-square (top-85 vs pool): chi2 = 111.0`; median fitted age top-85 = 36 y vs rest = 9 y. Compositional generalization: temperature contrast `low - high -0.497` (ctx → Decision, p < .0001); acquisition slope `-1.2801` |

Training is stochastic, so the values in steps 1–2 vary slightly from run to run (e.g. demo-network modularity ≈ 0.23, global efficiency ≈ 0.57, rich club ≈ 1.0); step 3 is deterministic.

**Expected run time** on a normal desktop (CPU only): about 4–6 minutes in total: step 1 ≈ 2–4 min, step 2 ≈ 1.5 min, step 3 ≈ 45 s.

## 4. Instructions for use

**Train a network with your own settings.** `run.py` trains a single network; all options are listed with `python run.py --help`. Key options:

- `--tasks`: tasks to train on (the study uses the 20 `poli.*` tasks listed in `demo/run_demo.sh`).
- `--foraging mvt` with `--forage-alpha-local`, `--forage-beta-global`, `--forage-travel-steps`, `--forage-temperature`, `--forage-eps`: MVT curriculum parameters (temperature controls how curiosity-driven vs. random task switching is).
- `--l1-weight` or `--distance-penalty --distance-path brain/assets/euclidean_distances_schaefer100.npy --distance-weight`: biophysical (L1 or wiring-cost) regularization.
- `--steps`, `--save-every`, `--logdir`, `--run-name`: training length, checkpointing, and output location (name runs `run_*` so the analysis scripts find them).

**Run a parameter sweep.** `experiments/run_sweep_forage.py --config forage_v6` defines the study's grid (see [`startup_demo.md`](startup_demo.md) §5 for local and SLURM submission).

**Analyse your own runs.** Point the extraction scripts at a folder of `run_*` directories:

```bash
python analysis/Synaptic_analysis/extract_weights.py --sweep_dir <folder> --output <weights.csv> --save_npy
python analysis/Network_analysis/compute_rnn_topology.py --sweep_dir <folder> --output <topology.csv>
python analysis/Performance_analysis/extract_performance.py --sweep_dir <folder> --output <performance.csv>
```

then aggregate (`analysis/aggregate_all_metrics.py`) and run the consolidated R scripts as in step 3 of the demo.

## Reproducing the main analyses

All reported figures and statistics come from three consolidated scripts, run from the project root after metrics are aggregated:

``` bash
Rscript analysis/Final_analyses_and_plots/pub_topology_consolidated.R     # network structure
Rscript analysis/Final_analyses_and_plots/pub_synaptic_consolidated.R     # synaptic density
Rscript analysis/compositional_analysis/pub_compositional_generalisation.R # compositional gen.
```

Each prints its statistics and writes PNGs to `analysis/Final_analyses_and_plots/Figures/`. "Most human-like" is defined everywhere by a single Mahalanobis fingerprint over the three topological metrics (`METRIC_COLS`, kept in sync across the consolidated scripts and the cohort builder).

Training, cluster sweeps, the per-run extraction steps, and the compositional probing are documented step-by-step in [`startup_demo.md`](startup_demo.md).

## Data

This repository ships the **shareable** data needed to reproduce everything downstream of the restricted inputs:

- `analysis/Cognitive_analysis/Data/Infant_Data/infant_foraging_release.csv` — anonymized infant foraging trials (no identifiers; integer `subj_id` only).
- Model-derived metric tables and the derived human topology table.

It **excludes** raw infant data / demographics / IDs and the HCP-derived connectomes, which cannot be redistributed. See [`DATA.md`](DATA.md) for the exact exclusions and how to obtain the restricted inputs (infant data: contact the authors under ethics/consent; HCP data: https://www.humanconnectome.org/).

## License

- Code: MIT License ([`LICENSE`](LICENSE)).
- Data: CC BY 4.0 ([`LICENSE-DATA`](LICENSE-DATA)).
- `Mod_Cog/` is third-party code (Khona et al.) released under CC0 ([`Mod_Cog/LICENSE`](Mod_Cog/LICENSE)).

## Citation

Please cite the accompanying manuscript and this repository ([`CITATION.cff`](CITATION.cff)).

# Project Instructions

## Bash Execution

- The VSCode extension silently rejects foreground Bash commands that run long (e.g., iterating over hundreds of files on the Z: network drive). Always use `run_in_background: true` for any command that processes many files or reads from Z:/fp02/.
- Python is not on PATH. Use the full path: `"C:/Users/fp02/AppData/Local/anaconda3/envs/ngym39/python.exe"`
- R is at: `"C:/Program Files/R/R-4.2.2/bin/Rscript.exe"`
- The working directory is already the project root. Do not prepend `cd` to commands.

## Data Paths

- Network drive (cluster data): `Z:/fp02/logs/sweep/forage_v5.1/forage_v6` (local mount of `/imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6`)
- Path resolution utility: `analysis/utils/paths.py` — use `get_logs_dir()` to auto-detect Z: vs cluster paths

## Cluster Nodes

- Only flag nodes as bad if they are DOWN/DRAIN/NOT_RESPONDING. High CPU load or full allocation is normal — the daemon handles busy nodes.
- Current bad nodes: node-k01, node-k12

## Sweep Design

- When designing parameter sweeps, do not remove existing parameter values — only add new ones. The user wants exhaustive combinatorial coverage.

## Sweep Script Responsibilities (migrated from .agent/rules/sweep-guide.md)

- `run_sweep_slurm.py` should only interact with the cluster: no decisions about model settings are made here — just check what's running and submit jobs.
- `run_sweep.py` is the only place where parameter values are set. They can be *free* (e.g. vary L1 weight and distance weight — varied here so `run.py` knows what to sweep) or *fixed* (e.g. target-sr = 1.0 for all models).
- `run.py` must hold only sensible defaults. It fills in the basic decisions that aren't set explicitly in `run_sweep.py`.

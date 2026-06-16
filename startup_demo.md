# University Cluster Setup (login-k01) 🎓

## 1. How to Link VS Code 🔗

- **Install Extension:** In your local VS Code, install "Remote - SSH" (Microsoft).
- **Connect:**
  1.  Press F1 (or Ctrl+Shift+P) -\> Type `Remote-SSH: Connect to Host...`
  2.  Enter: `fp02@login-k01` (or your configured host alias).
- **Result:** VS Code opens a new green window connected to the cluster. Now running on the cluster.

## 2. Check Disk Space (Crucial!) 💾

Before installing, check if you have space (PyTorch + CUDA needs \~5-7GB).

``` bash
df -h ~  # Check 'Avail' space in your home folder
```

**If you are low on space (\< 10GB):**

``` bash
# 1. Clean Conda Cache (Safest way to free GBs)
conda clean --all -y

# 2. Clean Pip Cache
rm -rf ~/.cache/pip
```

## 3. Environment Setup Commands 🛠️

Run these in the VS Code terminal (once connected) or your standard SSH terminal.

``` bash
# 1. Load Conda (Standard cluster step)
module load conda

# 2. Create Environment (Python 3.9)
conda create -n ngym39 python=3.9 -y

# 3. Activate
conda activate ngym39

# 4. Install Dependencies (Pinned Config)
pip install "gym==0.25.2" "numpy==1.23.5"

# 5. Mod_Cog & Neurogym (Clone & Install)
# We install in ~/neurogym (editable mode) to allow code modifications
cd ~
git clone https://github.com/neurogym/neurogym.git
cd neurogym
git checkout b060cc29dc36243e6d6b09ae480381ed1aa35207
pip install -e .
cd ..

# 6. Install PyTorch with CUDA (GPU)
# Using --no-cache-dir to prevent disk space errors
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 7. Install Helpers (Analysis & Progress)
pip install seaborn pandas scipy scikit-learn tqdm numba
```

**Network-analysis dependencies** (used by *Step 4: Compute Network Topology*). The upgraded topology metrics — Leiden modularity, weighted ensemble-normalized rich club, small-worldness (ω) and participation — plus the human consensus network need these on top of base `ngym39`:

``` bash
conda activate ngym39            # Step 4 now runs in ngym39 (cluster and local)
pip install bctpy leidenalg python-igraph   # graph metrics (both scripts)
pip install h5py netneurotools              # .mat loading + struct_consensus (human script only)
```

**Local R dependencies** (used by the final publication analysis — Step 10). The three consolidated `pub_*` scripts need:

``` r
# R >= 4.2 (Rscript at "C:/Program Files/R/R-4.2.2/bin/Rscript.exe")
install.packages(c("dplyr","tidyr","ggplot2","readr","scales",   # core
                   "emmeans","car","mgcv",                        # models / contrasts
                   "survival",                                    # AFT (compositional)
                   "plot3D"))                                     # 3D structural scatter
```

## 4. GitHub Project Setup

Your scripts are stored on the network drive:

- **Path**: `/imaging/astle/fp02/RNN`
- **Repo**: `https://github.com/FrancescPoli/multitask-curiosity.git`

### How to Update

``` bash
cd /imaging/astle/fp02/RNN
git pull origin main
```

## 5. Running Parameter Sweeps 🚀

### Preferred Method: Daemon Mode 🟢

"Daemon" mode ( Dynamic Job Manager) is smarter and safer for large sweeps. It acts as a background monitor that:

1.  **Scans for the best available nodes** before sending *each* individual job.
2.  **Waits if the cluster is full**, checking periodically until space frees up to send the next job.
3.  **Automatically submits** jobs to Slurm, so you don't need to run `sbatch`.

``` bash
# 1. Prepare Environment
module load conda
conda activate ngym39

# 2. Run Daemon (Scans + Submits automatically)
python experiments/run_sweep_slurm.py --sweep-id sweep_regs2 --mode daemon --strategy best --config default
```

You can also set a specific output folder with base dir:

``` bash
module load conda && conda activate ngym39

python experiments/run_sweep_slurm.py --sweep-id forage_v8 --config forage_v8  --base-dir /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 --mode daemon --strategy best
```

### Alternative: Basic Mode (Manual) ⚠️

"Basic" mode is simpler but less adaptive. It:

1.  Checks for the best nodes **only once at the beginning**.
2.  Generates a script to submit **all jobs immediately**.
3.  **Requires you to run `sbatch`** manually.

*Note: This is suitable only for a smaller number of jobs, as it doesn't adapt to changing cluster load.*

``` bash
# 1. Prepare Environment
module load conda
conda activate ngym39

# 2. Generate Submission (Creates submit_sweep.sbatch)
python experiments/run_sweep_slurm.py --sweep-id final_sweep_regs --mode basic --strategy best

# 3. Submit to Cluster
sbatch submit_sweep.sbatch
```

*(The script logs the config to `logs/sweep/final_sweep_v1/config_list.txt`)*

### Checking Job & Node Status 📊

``` bash
squeue -u fp02      # List your running jobs
scancel <JOB_ID>    # Cancel a specific job
scancel -u fp02     # Cancel ALL your jobs

# Check Node Health (e.g., why is node-j03 down?)
scontrol show node node-j03

# Update Running Job (e.g., move to low priority)
scontrol update JobId=2829417_[26-38] QOS=lopri

# Debug Job Failures: See which nodes caused failures
# Replace JOB_ID with your actual job array ID (e.g., 2966086)
sacct -j JOB_ID --format=JobID,NodeList,State,ExitCode -P | sort -t_ -k2 -n | head -100
```

## 6. Comparing Experiments 📈

After running a sweep, analyze and compare model performance across regularization types.

> **Note:** The old `compare_experiments.py` is for non-forage sweeps only. For forage sweeps, always use `compare_experiments_forage.py`.

``` bash
python analysis/compare_experiments_forage.py \
    --sweep_dirs /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --mode winners --compute-gen
```

**Arguments:**

- `--sweep_dirs`: Directory (or directories) containing trained model runs
- `--mode`: Analysis selection mode:
  - `winners`: Analyze best models only (recommended)
  - `all`: Analyze all models
  - `targeted`: Analyze specific runs (use with `--target-runs`)
  - `none`: Aggregate metrics only (no individual plotting)
- `--compute-gen`: Flag to run generalization probe/eval (slow) on selected models
- `--gen-method`: Method for generalization (`probe` = few-shot learning \[default\], `zero_shot` = direct evaluation)
- `--gen-tasks`: Override default 8 held-out Poli tasks (e.g., `--gen-tasks poli.go` for quick check)
- `--probe-steps`: Optimization steps for probe method (default: 2000)
- `--no-auto-analysis`: Skip automatic detailed analysis plots for winning models
- `--execution-mode`: `local` (default) or `cluster` (submits SLURM jobs)

**Default Behavior:**

- Evaluates on 8 held-out Poli tasks: `poli.antigo`, `poli.dlygo`, `poli.ctxgo`, `poli.dm1`, `poli.antidm1`, `poli.ctxdm1`, `poli.dlyantigo`, `poli.ctxdlydm1`
- Selects best model per regularization type based on **mean accuracy** across training tasks
- Generates comparison plots in `logs/sweep/[sweep_name]/comparison_plots/`

### Targeted Analysis 🎯

If you only want to analyze a specific set of runs (e.g., for debugging or looking at outliers), use `targeted` mode:

``` bash
python analysis/compare_experiments_forage.py \
    --sweep_dirs /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --mode targeted \
    --target-runs "91" "45" "102"
```

- `--mode targeted`: Activates filtering.
- `--target-runs`: List of strings to match.
- **Result:** Runs standard analysis/plotting on these runs. **Does NOT run probe**.

To **also run generalization probe** on these targets:

``` bash
python analysis/compare_experiments_forage.py \
    --sweep_dirs /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --mode targeted --compute-gen
```

### Parallel Analysis on Cluster ⚡

To speed up slow probing/analysis, you can offload the work to the cluster nodes:

``` bash
python analysis/compare_experiments_forage.py \
    --sweep_dirs /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --execution-mode cluster --mode winners --compute-gen
```

- **What it does**: Scans for "Best Models" and submits a separate SLURM job for each one to run the detailed analysis (probes, etc).
- **Workflow**:
  1.  Run with `--execution-mode cluster`.
  2.  Wait for jobs to finish (use `squeue -u fp02`).
  3.  Run again with `--execution-mode local` (default) to generate plots using the cached results.

## 7. SSH Setup (Optional) 🔐

**Why?** Set up SSH keys to avoid typing your password repeatedly when pushing/pulling from GitHub on the cluster.

**Steps:**

1.  **Generate SSH key on cluster:**

    ``` bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    # Press Enter to accept default location
    # Press Enter twice for no passphrase (or set one for security)
    ```

2.  **Copy public key:**

    ``` bash
    cat ~/.ssh/id_ed25519.pub
    ```

3.  **Add to GitHub:**

    - Go to GitHub → Settings → SSH and GPG keys → New SSH key
    - Paste the public key content

4.  **Test connection:**

    ``` bash
    ssh -T git@github.com
    # Should see: "Hi [username]! You've successfully authenticated..."
    ```

5.  **Update remote URL (if needed):**

    ``` bash
    cd /imaging/astle/fp02/RNN
    git remote set-url origin git@github.com:FrancescPoli/multitask-curiosity.git
    ```

6.  **Simply push changes**

    ``` bash
    git push
    ```

## 8. Run Comparisons (Synaptic, Network, Cognitive) 🧠

The full human-comparison pipeline has 7 steps. Steps 1–4 run on the **cluster** (heavy extraction). Steps 5–7 run **locally** (R analysis + aggregation). All extraction scripts support **resume** — safe to restart if interrupted.

> Steps 1, 2, and 3 are independent and can run in **parallel** (separate terminals). Step 4 depends on Step 1 completing.

### Step 1: Extract Weights (Synaptic & Network) — Cluster, ngym39

Extracts weight evolution (L1/L0 for Synaptic Space) and full weight matrices (for Network Topology) from model checkpoints.

``` bash
module load conda && conda activate ngym39

python analysis/Synaptic_analysis/extract_weights.py \
    --sweep_dir /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --save_npy
# Output:
#  - analysis/Synaptic_analysis/Data/population_weights.csv (Aggregated)
#  - [run_dir]/weights_evolution.csv (Per-run stats)
#  - [run_dir]/weights_evolution.npy (Full matrices, if --save_npy)
# Tip: Add --limit 5 for a quick test run
```

### Step 2: Extract Performance — Cluster, ngym39

``` bash
module load conda && conda activate ngym39

python analysis/Performance_analysis/extract_performance.py \
    --sweep_dir /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6
# Output: analysis/Performance_analysis/Results/performance_metrics.csv
```

### Step 3: Fit MVT Cognitive Metrics — Cluster, ngym39

Grid-searches over (alpha, beta) pairs to recover behavioural metrics. Slowest step.

``` bash
module load conda && conda activate ngym39

python analysis/Cognitive_analysis/fit_mvt_models.py \
    --sweep_dir /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --output analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv
# Output: analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv
```

**Alternative: submit as a SLURM array job** (8 parallel shards on compute nodes, \~8× faster):

``` bash
conda activate ngym39
sbatch analysis/Cognitive_analysis/fit_mvt_models.sh

# Once all 4 array tasks finish, merge the shard CSVs into a single file:
python analysis/Cognitive_analysis/fit_mvt_models.py \
    --sweep_dir /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --merge \
    --output analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv
```

The `conda activate` step is required — `sbatch` copies the submitting shell's environment to the compute node, and the job won't find conda otherwise.

Each array task writes to `model_mvt_params_fitted_shard{0..7}of8.csv`; the `--merge` step concatenates them into the final CSV. To change the shard count, edit both `#SBATCH --array=0-N` and `N_SHARDS` in the `.sh` file.

> ⚠️ The `.sh` file has the sweep dir, output CSV, and SBATCH log paths hardcoded (defaults point to `/imaging/astle/fp02/...`). Edit the defaults or pass positional args (`sbatch fit_mvt_models.sh <SWEEP_DIR> <OUT_CSV>`) if your paths differ.

### Step 4: Compute Network Topology — Cluster, ngym39

Requires `weights_evolution.npy` files from Step 1, and the network-analysis packages `bctpy leidenalg python-igraph h5py netneurotools` (see Part 3 — Environment Setup).

``` bash
module load conda && conda activate ngym39

# A. Human topology (ONE-TIME — only needs to run once, human data doesn't change)
python analysis/Network_analysis/extract_schaefer_metrics.py
# Output: analysis/Network_analysis/Results/human_topological_metrics.csv
# Note: this is the file consumed by aggregate_all_metrics.py and plot_structural_comparison.R.
# extract_hcpya_metrics.py is an older variant that uses a different parcellation — do not use.

# B. RNN topology (re-run for each new sweep)
python analysis/Network_analysis/compute_rnn_topology.py 
    --sweep_dir /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6
# Output: analysis/Network_analysis/Results/rnn_topological_metrics.csv
```

------------------------------------------------------------------------

**After transferring CSVs locally, run the remaining steps on your local machine:**

### Step 5: R Analysis (Synaptic, Network, Cognitive) — Local, R

``` bash
# Synaptic space: compare weight trajectories with human synaptic density
# Input: analysis/Synaptic_analysis/Data/population_weights.csv
# Output: analysis/Synaptic_analysis/Results/synaptic_metrics.csv
Rscript analysis/Synaptic_analysis/analyze_synaptic_space.R

# Network space (LEGACY figure — the paper's network figures/stats now come from
# pub_topology_consolidated.R in Section 9; keep only for a quick structural check):
# Input: rnn_topological_metrics.csv + human_topological_metrics.csv
# Output: analysis/comparison_plots/structural/structural_comparison_3d.png
Rscript analysis/Network_analysis/plot_structural_comparison.R

# Cognitive space: compare model MVT behavior with infants
# ONE-TIME: Fit infant baseline (only re-run if infant data changes)
Rscript analysis/Cognitive_analysis/fit_infant_mvt.R

# Compare models to infant baseline
Rscript analysis/Cognitive_analysis/analyze_cognitive_space.R \
    analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv
# Output: analysis/comparison_plots/cognitive/
```

### Step 6: Aggregate All Metrics — Local, Python

Merges synaptic, network, cognitive, and performance metrics. Computes normalized distances and grand_distance (L2 across subspaces).

``` bash
python analysis/aggregate_all_metrics.py
# Input: all CSVs from Steps 1-5
# Output: analysis/grand_unified_metrics_v2.csv
```

### Step 7: Chi-Square Winner Analysis — Local, R *(legacy / exploratory)*

Identifies which parameter values are enriched among top-performing models, using the legacy `norm_dist_*` distances.

``` bash
Rscript analysis/Forest_analysis/chisquare_analysis.R
# Input: analysis/grand_unified_metrics_v2.csv
# Output: analysis/Forest_analysis/Results/Chisquare/
```

> For the paper, the enrichment χ² and the fitted-age χ² are produced (on the Mahalanobis fingerprint) by the consolidated scripts in **Section 9** (`pub_topology_consolidated.R` and `pub_synaptic_consolidated.R`). Keep this step only for the broader `norm_dist_synaptic/cognitive/grand_distance` exploration.

### Step 8: Compositionality Probing — Cluster, ngym39

Tests whether the most human-like models (H-100) can solve held-out task combinations by learning only a new rule input vector, with all recurrent weights frozen. Compared against three matched control cohorts that vary exactly one or two parameters at a time.

**8A: Generate cohort ID lists — Local, Python**

Run the cohort builder to select the top-100 human-like models and generate three matched control cohorts (C-temp, C-reg, C-both):

``` bash
# Reads grand_unified_metrics_v2.csv + human_topological_metrics.csv, selects
# H-100 by Mahalanobis distance to the HCP-YA average over the three topological
# metrics (modularity_leiden, efficiency, rich_club) — the SAME fingerprint as
# pub_topology_consolidated.R — then applies the structural filter
# (temp < 0.01 & reg_value > min) and finds matched controls for each H-100 model.
# Outputs: analysis/compositional_analysis/Data/cohorts/{h100,c_temp,c_reg,c_both}_ids.txt + all_ids.txt + cohort_map.csv
python analysis/compositional_analysis/build_compositionality_cohorts.py
```

H-100 selection: most human-like by the Mahalanobis topological fingerprint (`METRIC_COLS` in the script — keep in sync with `pub_topology_consolidated.R`). Filter rationale: `temp < 0.01` keeps only deterministic switchers; `reg_value > min` excludes the near-zero regularization edge cases that cannot have a meaningful C-reg control. A double performance filter is applied: `fraction_solved >= 0.3` in the broad pool and `>= 0.9` within the top-100.

> Re-selecting with the Mahalanobis fingerprint changes cohort membership vs the legacy `norm_dist_network` selection. Diff the new `compositional_analysis/Data/cohorts/all_ids.txt` against models that already have `probe_compositionality.json` in the sweep, and probe **only the new IDs** (the rest are reusable).

Control cohorts (each model in H-100 has a matched counterpart with exactly one/two params changed): - **C-temp**: `temp → 1.0` — random switching, all other params identical - **C-reg**: `reg_value → minimum` (L1: `1e-06`, Distance: `1e-05`) — minimal pruning pressure - **C-both**: both changes simultaneously — fully non-human-like baseline

**8B: Push and run probes — Cluster**

``` bash
# Sync the repo so the cluster has the latest scripts and cohort files
git push

# On login-k01:
module load conda && conda activate ngym39

# Run all four cohorts (independent — can submit in parallel across terminals)
# The script scans the sweep dir once to build its index, then submits one SLURM job
# per model. Results are written to probe_compositionality.json in each run directory.
# Copy the exact commands printed by build_compositionality_cohorts.py, e.g.:

python analysis/compare_experiments_forage.py \
    --sweep_dirs /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --mode targeted \
    --target-runs $(cat analysis/compositional_analysis/Data/cohorts/h100_ids.txt | tr '\n' ' ') \
    --compute-gen --gen-set full --probe-steps 5000 \
    --execution-mode cluster

# Repeat for c_temp_ids.txt, c_reg_ids.txt, c_both_ids.txt
# (or just probe analysis/compositional_analysis/Data/cohorts/all_ids.txt — or only the IDs that don't yet
#  have probe_compositionality.json, to skip the ones already probed)

# Check job progress
squeue -u fp02
```

`--gen-set full` probes all 32 tasks: 20 training controls (input-mapping baseline) + 12 held-out compositional targets across three transport groups. `--gen-set quick` probes only 6 tasks (3 training + 3 held-out) for rapid iteration.

**8C: Collect results — Local**

Once all SLURM jobs finish, re-run with `--execution-mode local` to load cached results and generate plots:

``` bash
python analysis/compare_experiments_forage.py \
    --sweep_dirs /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --mode targeted \
    --target-runs $(cat analysis/compositional_analysis/Data/cohorts/all_ids.txt | tr '\n' ' ') \
    --compute-gen --gen-set quick \
    --execution-mode local
# Output: comparison_plots/ inside the sweep dir
# Use analysis/compositional_analysis/Data/cohorts/cohort_map.csv to split results by cohort for statistics
```

**8D: Build dataset and run statistical analysis — Local**

Once all SLURM probe jobs have finished, build the analysis-ready dataset and run the R analysis:

``` bash
# 1. Build compositional_dataset.csv (one row per probed model, ~110 rows)
#    Also writes task_metadata.csv (32 tasks × modifier flags)
#    Input:  analysis/grand_unified_metrics_v2.csv
#            analysis/compositional_analysis/Data/cohorts/cohort_map.csv
#            Z:/fp02/logs/sweep/forage_v5.1/forage_v6/*/probe_compositionality.json
#    Output: analysis/compositional_analysis/Data/compositional_dataset.csv
#            analysis/compositional_analysis/Data/task_metadata.csv
python analysis/compositional_analysis/build_compositional_dataset.py

# 2. Extract base-ability accuracy covariate (end-of-training accuracy on the
#    base tasks the held-out recombinations redeploy). Reads run_curriculum.csv
#    from the sweep (Z:); required by the AFT model below as its z_acc covariate.
#    Run via PowerShell so Z: is visible.
#    Input:  analysis/compositional_analysis/Data/compositional_dataset.csv (run list)  +  sweep run_curriculum.csv
#    Output: analysis/compositional_analysis/Data/compositional_base_accuracy.csv
python analysis/compositional_analysis/extract_base_accuracy.py \
    --sweep_dir /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6
```

**Canonical analysis (reproduces the Methods statistics).** The reported β's come from a log-normal accelerated-failure-time model on the geometric-mean solve time `G = sqrt(t_reference * t_held-out)`:

``` bash
# Input:  analysis/compositional_analysis/Data/compositional_dataset.csv
#         analysis/compositional_analysis/Data/compositional_base_accuracy.csv  (z_acc covariate)
# Model:  Surv(G, both_solved) ~ temperature * regularisation * recombination + z_acc
# Output: analysis/Final_analyses_and_plots/Figures/pub_compositional_generalisation.png
#         Console: interaction LRTs; temperature|recombination & regularisation contrasts
"C:/Program Files/R/R-4.2.2/bin/Rscript.exe" \
    analysis/compositional_analysis/pub_compositional_generalisation.R
```

`temp`/`reg` are derived from cohort membership (low-τ = {H-100, C-reg}, strong-reg = {H-100, C-temp}); `grp` is the recombination type (ctx→Decision, anti→Match, ctx+anti→Match).

> All compositional-generalization scripts live in `analysis/compositional_analysis/` and read/write their data under `analysis/compositional_analysis/Data/` (incl. the `cohorts/` and `cohorts_old_norm_dist/` folders). The Fig-4 solve-time scatter is `pub_comp_scatter.R` in the same folder.

## 9. Publication Figures & Statistics 📊

All figures and statistics reported in the Methods come from **three consolidated `pub_*` scripts**, run locally after Step 6 (and Step 8 for compositional). Each is self-contained and driven by one shared knob — `METRIC_COLS` — which **must be identical across all three and the cohort builder** (currently `modularity_leiden, efficiency, rich_club`). "Most human-like" everywhere = smallest Mahalanobis distance to the HCP-YA individual average over those three metrics.

``` bash
RS="C:/Program Files/R/R-4.2.2/bin/Rscript.exe"

# A. NETWORK STRUCTURE — 3D scatter, Mahalanobis similarity regression
#    (full-τ factorial + 4-hyperparameter robustness), per-metric, enrichment χ².
#    Input: grand_unified_metrics_v2.csv + human_topological_metrics.csv
& $RS analysis/Final_analyses_and_plots/pub_topology_consolidated.R

# B. SYNAPTIC DENSITY — human GAM, density trajectories, fitted-age landscape + χ²,
#    accuracy~age GAM. (age_cap is preprocessing: Step 5 analyze_synaptic_space.R.)
#    Input: grand_unified_metrics_v2.csv + human synapse data + synaptic_metrics.csv
& $RS analysis/Final_analyses_and_plots/pub_synaptic_consolidated.R

# C. COMPOSITIONAL — log-normal AFT on geometric-mean solve time (see Step 8D).
#    Lives with the rest of the compositional pipeline, reads from its Data/ folder.
#    Input: compositional_analysis/Data/{compositional_dataset,compositional_base_accuracy}.csv
& $RS analysis/compositional_analysis/pub_compositional_generalisation.R
```

Each prints the exact statistics that appear in the Methods (regression/ANOVA terms, post-hoc contrasts, χ², GAM edf/F, AFT β's) and writes its `pub_*.png` figures into `analysis/Final_analyses_and_plots/Figures/`. (The network and synaptic consolidated scripts live in `analysis/Final_analyses_and_plots/`; the compositional one lives with its pipeline in `analysis/compositional_analysis/`.)

> Legacy/exploratory scripts that these supersede for the paper: `plot_structural_comparison.R` and `chisquare_analysis.R` (Steps 5/7) and `analyse_compositionality.R` (Step 8D). The many root-level `tmp_*.R/.py` files are scratch explorations (alternative model specs, robustness checks, the Goldilocks question) and are **not** part of the reported pipeline. The base-ability accuracy covariate, formerly extracted by the root-level `tmp_train_acc.py`, is now produced by the proper pipeline step `analysis/Cognitive_analysis/extract_base_accuracy.py` (Step 8D).

## 10. Syncing Cluster and Local Work 🔄

### Part 1: On The Cluster (VS Code Window)

**Objective:** Save your work and ship it out on a "lifeboat" branch.

**Prerequisite (if your push fails with "RPC failed"):** Run this in the terminal: `git config http.postBuffer 1048576000`

**Commit:**

1.  Go to Source Control sidebar.
2.  Enter message "WIP Cluster" and click Commit.

**Create Branch:**

1.  If cluster-sync branch doesn't exist, click the three dots for RNN Git, then select "Create new branch...".
2.  Name it `cluster-sync`.

**Push:**

1.  Click the Cloud Icon (Publish Branch) in the Source Control sidebar.

### Part 2: On Local Machine (VS Code Window)

**Objective:** Save local work, grab the cluster work, and combine them.

**1. Check your branch:** Ensure the bottom-left corner says `antigravity`.

**2. Save Local Work:** Go to Source Control and **Commit** (message: "WIP Local").

**3. Fetch & Merge (The Easy Way):**

1.  Click the **three dots (...)** in Source Control -\> **Fetch**.
2.  Click the **three dots (...)** -\> **Branch** -\> **Merge Branch...**.
3.  Select `origin/cluster-sync`.
4.  If conflicts appear, use the UI "Resolve in Merge Editor" to pick which code to keep.

**4. Push (The Sync Step):** Click **Sync Changes** (bottom-left) to send the combined code back to GitHub.

------------------------------------------------------------------------

#### 💡 Advanced: Selective Merge (Selective Folders)

Use this in the terminal if you *only* want specific folders (like `logs/` or `experiments/`) from the cluster and want to ignore all other cluster code changes:

1.  `git merge origin/cluster-sync --no-commit --no-ff`
2.  `git checkout HEAD -- .` (keep all local code)
3.  `git checkout origin/cluster-sync -- logs/ experiments/` (grab specific cluster folders)
4.  `git commit -m "Merged specific cluster folders"`

------------------------------------------------------------------------

### Part 3: On The Cluster (VS Code Window)

**Objective:** Get back to the main branch and pull the final result.

**1. Switch Back to Main Branch:**

1.  Click the branch name (`cluster-sync`) in the bottom-left corner.
2.  Select `antigravity` from the list.

**2. Pull updated code:** Click **Sync Changes** (rotating arrows, bottom left).

**3. Prepare for next time (Optional):** If you want to continue working on the cluster and keep using the same "lifeboat", merge the updated `antigravity` back into your `cluster-sync` branch:

1.  Switch back to `cluster-sync`.
2.  Click **three dots (...)** -\> **Branch** -\> **Merge Branch...** -\> Select `antigravity`.

**Done!** Both VS Code windows are now perfectly in sync on the `antigravity` branch.
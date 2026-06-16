# Human Brain Comparison Strategy

---

## Required Fixes

### 1. Grand Distance: Normalize Subspace Distances + Use L2
**Current (`aggregate_all_metrics.py`):**
```python
grand_distance = dist_synaptic + dist_network + dist_cognitive
```
**Two problems:**

**A. Incomparable units across subspaces.** Synaptic distance is computed in raw fraction space (peak_fraction, pruning_fraction in [0,1], correlation in [-1,1]). Network and Cognitive distances are computed in z-score space (units = number of human/infant SDs). These scales are fundamentally different — whichever subspace produces larger typical distances dominates the grand ranking, making the other subspaces nearly irrelevant.

**B. L1 sum instead of L2.** Summing distances has no clean geometric interpretation and further amplifies scale imbalance.

**Fix:** Two-step correction:
1. **Normalize each subspace distance to [0,1]** by dividing by the 95th percentile of that distance across the population (clamped to 1.0). This is robust to outliers, preserves relative magnitude within each space, and ensures each subspace contributes equally by default.
2. **Combine with L2** (Euclidean) instead of L1:
```python
q95_syn = dist_synaptic.quantile(0.95)
q95_net = dist_network.quantile(0.95)
q95_cog = dist_cognitive.quantile(0.95)

norm_syn = (dist_synaptic / q95_syn).clip(upper=1.0)
norm_net = (dist_network / q95_net).clip(upper=1.0)
norm_cog = (dist_cognitive / q95_cog).clip(upper=1.0)

grand_distance = sqrt(norm_syn**2 + norm_net**2 + norm_cog**2)
```
Each normalized sub-distance lives in [0,1], so grand_distance ranges from 0 (perfect match on all three) to sqrt(3) ≈ 1.73 (worst on all three). **Implemented** in `aggregate_all_metrics.py`.

### 2. Synaptic Normalization: Replace Hardcoded Denominators with Dimensionless Metrics
**Current (`analyze_synaptic_space.R`):**
```r
d_p <- ((human_peak_age - curr_peak) / 40)^2
d_d <- ((human_decay_rate - curr_decay) / 0.1)^2
```
**Problem:** The three synaptic axes (correlation, peak age, decay rate) are not on comparable scales. Correlation is naturally in [0,1], but peak age is in years (range ~0–75) and decay rate is in normalized-density/year (range ~0–0.05). The hardcoded denominators /40 and /0.1 are ad-hoc approximations of the model range, not anchored to biological variability. As a result, the decay axis contributes at most ~0.1 to the squared distance even for a model with no pruning at all — it is nearly irrelevant.

**Fix:** Replace peak age and decay rate with dimensionless metrics that are naturally bounded in [0,1], eliminating the need for any denominator:
- **Peak Age (%)** = `peak_age / age_cap` — at what fraction of the modelled lifespan does the weight peak occur?  Human reference: `human_peak_age / 75 ≈ 4.3%` (using 75 years as the max candidate lifespan).
- **Total Pruning (%)** = `(peak_val − final_val) / peak_val` — what fraction of peak weight is lost by the end of training?  Human reference: computed from the Huttenlocher GAM curve.

The distance formula becomes:
```r
d_r <- (1 - curr_cor)^2
d_p <- (human_peak_fraction - curr_peak_fraction)^2
d_d <- (human_pruning_fraction - curr_pruning_fraction)^2
dist <- sqrt(d_r + d_p + d_d)
```
All three axes are now in [0,1] with no denominator. **Implemented** in `analyze_synaptic_space.R` and `synaptic_space_comparison.R`.

### 3. Tau Ratio: Replace Configured with Behaviorally Fitted Value
**Current (`fit_mvt_models.py` line 96):**
```python
tau_ratio = params.get('alpha', np.nan) / params.get('beta', np.nan)
```
**Problem:** `stickiness` and `R_noise` are both *behaviorally fitted* from model leave/stay events (via logistic regression on `run_curriculum.csv`). `tau_ratio` alone uses the *configured* hyperparameters from the run directory name. For infants (`fit_infant_mvt.R`), all three metrics — stickiness, R_noise, and tau_ratio — are jointly inferred from behavioral data using a MLE grid search over (alpha, beta). The inconsistency: for models we measure tau_ratio from what we *set*, while for infants we measure it from what we *observe*. This is not a neutral choice — if a model's effective timescale separation (as reflected in its switching behaviour) differs from its configured alpha/beta (e.g. due to signal clipping or curriculum effects), we are measuring the wrong thing.

**Fix:** In `fit_mvt_models.py`, add a grid search over (alpha, beta) pairs applied to the model's `run_curriculum.csv` data — using the same EMA-then-GLM logic as `fit_infant_mvt.R`. Use the best-fit (alpha, beta) to compute a behaviourally fitted tau_ratio. The existing GLM (for b0/b1) can be embedded within this grid search, making the full cognitive fit unified.

**Also affects:** `analysis/Cognitive_analysis/extract_mvt_metrics.py` line 103 has the same issue (uses configured alpha/beta directly). This is the older/simpler pipeline — verify whether it is still used downstream before fixing.

---

## Goal
Identify which Artificial Neural Network (ANN) models best resemble human intelligence by evaluating them across two fundamental spaces:
1.  **Brain Space (Neural Implementation):** Structural and developmental properties.
2.  **Cognitive Space (Behavior/Function):** Behavioral patterns, task switching, and curiosity-driven exploration.

---

## Model Parameters

All models use an MVT (Marginal Value Theorem) foraging controller for curriculum selection (see `curiosity/train/foraging.py`). The stay/leave decision each step is: **stay if P > rho + eps**, where P is local task reward and rho is the global reward baseline. With temperature > 0, this becomes stochastic via sigmoid: `prob_stay = σ((P − rho − eps) / temp)`.

| Parameter | Sweep values (V8) | Role |
|-----------|-------------------|------|
| **beta** | 0.0001, 0.001, 0.003, 0.01 | EMA rate for global reward baseline (rho). Small = slow-changing reference |
| **alpha** | 0.01, 0.03, 0.1, 0.2 | EMA rate for local task progress (P). Large = reactive to recent reward |
| **temp** | 0.0001, 0.001, **0.003**, **0.005**, 0.01, 0.1, 1.0 | Stochasticity of stay/leave decision. At temp→0: deterministic threshold; at high temp: near-random |
| **reg_type** | L1, Distance | Regularization mechanism (weight sparsity vs anatomical constraint) |
| **reg_value** | 0, 1e-5, 1e-4, 5e-4 | Regularization strength |
| **travel** | 50, 500 | Zero-reward steps incurred when leaving a task (switching cost) |
| **eps** | 0.0, -0.01 | Threshold margin. Negative = stickier (harder to leave current task) |

Parameters group into three mechanistic families:
* **Foraging dynamics**: beta × alpha × temp — how the agent decides when to leave a task
* **Regularization**: reg_type × reg_value — constraints on weight structure
* **Exploration cost**: travel × eps × temp — what switching costs and how sticky the policy is

---

# Part 1: Brain Space (Neural Implementation)

We evaluate models along three primary axes of comparison with the human brain's physical substrate:

### 1. Structure
Comparison of the network's static architecture and connectivity patterns. We focus on three maximally different topological metrics to capture distinct dimensions of network organization:

#### A. Key Metrics
1.  **Modularity (Segregation):**
    *   *What it measures:* The degree to which the network divides into distinct communities with dense internal connections.
    *   *Why selected:* It captures **Local Specialization**. High modularity indicates the network has specialized functional areas that operate somewhat independently.
2.  **Global Efficiency (Integration):**
    *   *What it measures:* The average inverse shortest path length between all node pairs.
    *   *Why selected:* It captures **Global Communication**. High efficiency means information can travel easily across the entire network, rewarding shortcuts that tie distal parts together.
3.  **Rich Club Coefficient (Hierarchy):**
    *   *What it measures:* The tendency of high-degree "hubs" to connect to each other, forming a central backbone.
    *   *Why selected:* It captures **Structural Hierarchy**. A "rich club" ensures the network has a robust core for integrating information, a key signature of mammalian brains.

#### B. Pruning Strategy (Normalization)
To compare Graph Theory metrics fairly between Artificial Neural Networks (ANNs) and Biological Brain Networks (which may have different baseline connectivities), we must normalize their density.
*   **Method:** **Proportional Thresholding (PT)**.
*   **Rule:** We prune both networks to **20% density** (retain the top 20% strongest connections).
*   **Edge Case:** If either network has a native density *below* 20%, we prune **both** to the level of the least dense network to ensure an exact match in the number of edges. This isolates topological differences from density differences.

### 2. Dynamics
Comparison of the network's temporal activity and state-space trajectories.
*   **Metrics:** Representational Similarity Analysis (RSA), Dimensionality of activity (PCA/Manifold).

### 3. Development (Implemented)
Comparison of how the network's properties evolve over "time" (training steps) versus human developmental stages.
*   **Key Comparison:** **Synaptic Density vs. Total Sum of Weights**
    *   **Biological Ground Truth:** Human synaptic density peaks in early childhood and then prunes down during adolescence/adulthood (inverted-U or decreasing curve depending on region).
    *   **Model Metric:** Track `Total Sum of Abs Weights` ($\sum |W|$) or `L0 Norm` (count of non-zero weights) throughout training.
    *   **Hypothesis:** Models with regularization (L1/Distance) might exhibit "pruning" behavior similar to biological development.

## Implementation Steps
1.  **Metric Implementation:** Create analysis scripts to extract these time-series metrics from model checkpoints.
2.  **Ground Truth Data:** Curate representative human/biological data curves for comparison.
3.  **Correlation Analysis:** Quantify the similarity between model trajectories and biological trajectories.

## Preprocessing: Alignment Methods

### 1. Binning / Resampling (Simple)
*   **Method:** Divide the timeline into $N$ equal bins and average data points.
*   **Pros:** Simple, fast.
*   **Cons:** Lose resolution; arbitrary bin edges.

### 2. GAM Fitting (Advanced)
Fit a **Generalized Additive Model (GAM)** or Spline to both datasets.
*   **Method:**
    1.  Fit Human Data: $Synapses(t) \approx f_{human}(t)$ (e.g., using `mgcv` in R or `pygam` in Python).
    2.  Fit Model Data: $Weights(t) \approx f_{model}(t)$.
    3.  Compare the continuous functions $f_{human}$ and $f_{model}$ (or their derivatives).
*   **Pros:**
    *   Handles uneven sampling perfectly (sparse biological data vs dense model logs).
    *   Provides smooth derivatives (rate of change), allowing us to compare **Peak Maturation Age**.
    *   Robust to noise.

## Adjudication: 3D Euclidean Distance (Implementation)

To select the best model fit **mathematically** without replying on visual adjudication, we optimize the **Lifespan Mapping** by minimizing the distance to the Human Target in a 3D metric space.

The metric space is defined by three normalized axes:
1.  **Correlation ($r$)**: Shape similarity. Target: $r=1.0$.
2.  **Synaptogenesis Peak ($P$)**: Age of maximum synaptic density (from Huttenlocher data). Target: Human Peak (~3.24 years).
3.  **Pruning Rate ($D$)**: Maximum negative slope during pruning phase. Target: Human Rate (~0.0310).

We calculate the **Normalized Euclidean Distance** ($d$) for each candidate lifespan ($L$):

$$ d(L) = \sqrt{ (1 - r_L)^2 + \left(\frac{P_{human} - P_L}{\max(P)}\right)^2 + \left(\frac{D_{human} - D_L}{\max(D)}\right)^2 } $$

The **Best Fit Model** is selected as:
$$ L^* = \arg\min_L d(L) $$

This method ensures we select models that not only "look" right (high correlation) but also match the key developmental milestones (peak timing and pruning speed) of the human brain.

---

# Part 2: Cognitive Space (Behavior/Function)

We evaluate whether models exhibit similar **exploration and task-switching dynamics** as human infants.

## Infant Foraging Data
Infants in a "foraging" task (e.g., free play with multiple toys/screens) exhibit specific behavioral dynamics:
*   **Engagement:** They focus on a stimulus/task to learn from it.
*   **Disengagement:** They leave the current task when information gain drops (or expected gain elsewhere is higher).
*   **Metric:** We track **Learning Progress (LP)** or information gain over time and correlate it with **Leaving Events**.

## Marginal Value Theorem (MVT) Analysis
We model this behavior using the **Marginal Value Theorem (MVT)**.
*   **Hypothesis:** An agent leaves a "patch" (task) when its instantaneous rate of return ($P$, local LP) drops below the average rate of return of the environment ($\rho$, global LP).
*   **Prediction:** $P(leave)$ increases as $(P - \rho)$ becomes negative.

### Current Implementation Critique (`infant_curiosity/MVT_analysis.R`)
We must rigorously verify the MVT analysis logic before applying it to models. Potential issues identified:

1.  **Scaling Logic:**
    *   *Current:* Uses `glmer(event ~ scale(relP) ...)`.
    *   *Problem:* `scale()` centers the data (mean=0, sd=1). The MVT hypothesis is about an **absolute threshold**: leave when $P < \rho$ (i.e., $relP < 0$).
    *   *Result:* If a subject is generally happy ($P > \rho$ most of the time), `scale(relP)` will force positive values below the mean to become negative. The GLM might then learn to predict "Leave" even when $P > \rho$, violating the core theory. We must use raw difference or uncentered scaling.

2.  **Parameter Fitting vs. Fixing:**
    *   *Current:* Fixed parameters `alpha_local` (halflife=1 step) and `beta_global` (halflife=11 steps).
    *   *Critique:* This assumes a specific "memory" implementation for all infants### 3. The Comparison Solution: Bounded Behavioural Metrics
Instead of comparing raw parameter values (which may differ in scale between biological and artificial agents), we compare their **functional role** in the decision process. We define a 3D "Cognitive Space" using three **bounded [0,1] metrics** recovered from behaviour via grid-search, mirroring the synaptic approach where all axes are naturally bounded:

| Axis | Formula | Range | What it captures |
|------|---------|-------|-----------------|
| **baseline_stay** | sigmoid(−b0) | [0,1] | Default tendency to stay when reward signal is neutral |
| **reward_modulation** | P(stay\|+1SD) − P(stay\|−1SD) | [0,1] | How much reward history drives stay/leave decisions |
| **timescale_balance** | α_hl / (α_hl + β_hl) | [0,1] | Relative learning speed (local vs global EMA) |

Key properties:
- **baseline_stay**: 0.5 = unbiased, >0.5 = sticky, <0.5 = restless. Sigmoid transform of GLM intercept b0 into probability space.
- **reward_modulation**: A random switcher (b1≈0) gets ≈0; a perfect MVT agent gets ≈1. Integrates both b1 AND signal_sd — captures whether reward sensitivity *matters* for this agent's behaviour.
- **timescale_balance**: 0.5 = equal timescales; →0 = fast local tracking; →1 = fast global adaptation. For models, alpha/beta are recovered via grid search (not configured), making this a true behavioural measure.

**Grid-search recovery (`fit_mvt_models.py`):** For both models and infants, we search over (alpha, beta) pairs, recompute EMA signals from raw rewards, fit a GLM at each grid point, and select the pair minimising AIC. This makes all three axes fully comparable: both sides use behaviourally recovered parameters.

**Distance:** Raw Euclidean to infant group mean — no z-scoring, no SD-based normalisation. Same approach as synaptic space:
```
dist_cognitive = sqrt((bs_model−bs_infant)² + (rm_model−rm_infant)² + (tb_model−tb_infant)²)
```

> **Historical note:** Earlier versions used unbounded metrics (R_noise ∈ orders of magnitude, stickiness b0 ∈ (−∞,+∞), tau_ratio from configured params). These were replaced because (1) unbounded scales made z-scoring unreliable, and (2) tau_ratio was not a behavioural property but a trivial function of sweep settings.

## Bridging Infants and Models
To connect the two:

11. **Equivalent Signals:**
    *   **Infants:** $D$ (Learning Progress / Surprise / Entropy drop). Computed as KL divergence of consecutive distributions.
    *   **Models:** $L_{t} - L_{t+1}$ (Loss drop), or explicit "Intrinsic Reward" signal.

2.  **Unified Analysis Pipeline:**
    *   Step 1: **Infant Fitting (MLE):**
        We fit the *exact same* MVT decision rule as used in `foraging.py` to the infant data $(r_t, \text{event}_t)$ using Maximum Likelihood Estimation.
        The decision rule is:
        $$ P(\text{leave}_t) = \sigma\left( \frac{\epsilon + \rho_t(\beta) - P_t(\alpha)}{T} \right) $$
        *   **Algorithm:** Grid search over $(\alpha, \beta)$ pairs. For each pair, compute the time-series $P_t, \rho_t$ and maximize the likelihood of the observed leave/stay sequence to find the best-fit $T$ (noise) and $\epsilon$ (bias).
        *   **Result:** Fitted $\alpha_{inf}, \beta_{inf}, T_{inf}, \epsilon_{inf}$.

    *   Step 2: **Model Parameters:** We already have these from the sweep configuration ($\alpha_{mod}, \beta_{mod}, T_{mod}, \epsilon_{mod}$).

    *   **The Comparison Solution: Relative Terms**
        Direct comparison is impossible (seconds vs steps). We compare **Dimensionless Ratios**:

        1.  **Timescale Separation (Alpha/Beta):**
            $$ R_{\tau} = \frac{\alpha}{\beta} $$
            *   *Meaning:* Neural/Process Memory. How much faster is local integration than global averaging? (e.g., "10x separation" vs "100x separation").

        2.  **Relative Noise (Inverse SNR):**
            $$ R_{noise} = \frac{T}{\sigma_{(P-\rho)}} $$
            *   *Meaning:* **Decision Determinism.** It measures the magnitude of decision noise ($T$) relative to the natural variation of the interest signal ($\sigma_{(P-\rho)}$).
            *   If $R_{noise} \ll 1$: Behavior is deterministic (only leave when signal drops).
            *   If $R_{noise} \gg 1$: Behavior is random (signal changes are swamped by noise).
            *   This allows us to compare "noisiness" even if infants have $T_{infant}$ (bits) and models have $T_{model}$ (nats).

        3.  **Stickiness (Total Inertia):**
            $$ \beta_0 \approx \frac{\epsilon_{total}}{T} $$
            *   *Meaning:* **Resistance to Switching.** This single parameter captures *all* friction against disengaging (Time Cost + Inertia).
            *   **Infant:** Fittted Intercept $\beta_0$ from GLM (Log-odds of leaving when $P=\rho$).
            *   **Sign Interpretation:**
                *   **Negative $\beta_0$:** **"Sticky"** (Low pr. of leaving when $P \approx \rho$).
                *   **Positive $\beta_0$:** **"Flighty/Picky"** (High pr. of leaving unless $P \gg \rho$).
            *   **Model:** Calculated as $\frac{\epsilon - \text{CostEquiv}}{T}$ (Note: signs depend on specific rule implementation, we calibrate to match Infant $\beta_0$).

## 6. Model Selection & Winner Analysis

Once models have been evaluated across the various distance metrics (e.g., Synaptic, Network, Cognitive, and Grand Distance), we can identify which parameters drive success.

### Winners Frequency Visualization
To conceptualize what makes a "winning" model, we isolate the top $K$ performing models (for example, the Top 50) for each distance metric and compare their parameter distributions against the base population of the entire sweep. 

- **Chi-Square Testing**: We use Goodness-of-Fit tests to identify parameters where the distribution inside the Top-K models differs significantly ($p < 0.05$) from the full sweep distribution.
- **Normalized Enrichment**: We visualize these distributions using stacked bar plots. Crucially, the size of each stack segment is mapped to its *Relative Selection Power* (normalized enrichment: $E / \sum E$), rather than its raw frequency. This effectively cancels out any base-rate imbalances in the original sweep, showing exactly how aggressively the winning parameter level dominated its alternatives in driving success.

### Winner Age Distributions (Synaptic & Network Landscapes)
We can also look at the distribution of the **best fitted age** (from the synaptic exponential decay fits) across the "winner models." By applying Chi-Square tests to this distribution versus the baseline population, we can see if a specific age (e.g., 5 years vs 25 years) dynamically "stands out" or dominates for the best performing models. We will eventually apply this exact same landscape analysis to the **Network space**.

---

## 7. Next Steps

### Background: What the Chi-Square Analysis Tells Us

Before defining the next experiments it is worth reading out what the winner analysis already established. The grand-distance Top 50 are not a random sample — they are sharply concentrated in a very small corner of parameter space:

| Parameter | Enriched levels | Depleted / Absent |
|---|---|---|
| `temp` | 1e-4 (1.7x), 0.001 (2.2x), 0.01 (2.1x) | **0.1 and 1.0 fully absent** |
| `travel` | 50 (2.2x) | **500 fully absent** |
| `reg_value` | 5e-4 (1.6x), 1e-4 (1.3x) | 1e-6 strongly depleted (0.4x) |
| `reg_type` | Distance (1.3x) | L1 mildly depleted |
| `alpha` | 0.03 (1.6x) | — |

In plain language: **the most brain-like models are the ones that switch deterministically (low temp), pay little travel cost, and maintain strong anatomical regularization**. High-temperature (random) switching and minimal regularization are both incompatible with human-like dynamics. The cognitive sub-space (MVT behavior) and the grand distance both replicate this pattern independently, while the synaptic sub-space is primarily driven by L1 + specific reg_value, largely independent of temp.

This gives us a natural causal decomposition to test: are these same parameter families responsible for **compositional generalization** — a property we care about beyond just human-similarity?

---

### Step 8: Define the H-50 Cohort and Matched Controls

#### 8A: The Human-Like Cohort (H-50)

Extract the 50 run IDs with lowest `grand_distance` from `analysis/grand_unified_metrics_v2.csv`. These are the models whose synaptic development, network topology, and cognitive foraging dynamics simultaneously resemble human biology. Call this **H-50**.

```python
import pandas as pd
df = pd.read_csv("analysis/grand_unified_metrics_v2.csv")
df = df[df.fraction_solved >= 0.6]
h50 = df.nsmallest(50, "grand_distance")[["run_id", "beta", "alpha", "temp", "reg_type", "reg_value", "travel", "eps", "grand_distance"]]
```

#### 8B: Three Matched Control Cohorts

For each H-50 model, we find its **counterfactual** in the sweep — an otherwise-identical run where exactly one or two parameters are replaced with a "non-human-like" value. This exploits the exhaustive combinatorial structure of the sweep.

| Cohort | Change | Mechanistic question |
|---|---|---|
| **C-temp** | `temp → 1.0` (all others identical) | Does deterministic exploration drive compositionality? |
| **C-reg** | `reg_value → 1e-06` (all others identical) | Does synaptic pruning / regularization drive compositionality? |
| **C-both** | `temp → 1.0` **and** `reg_value → 1e-06` | Fully non-human baseline: are the effects additive? |

Controls can be looked up directly in `grand_unified_metrics_v2.csv` by matching on `(beta, alpha, reg_type, travel, eps)` with the altered values of `temp` / `reg_value`. Since the sweep is fully factorial, matched controls will exist for most H-50 models.

> **Note:** "reg_value = 1e-06" is the near-zero regularization level (weakest pruning signal). "temp = 1.0" corresponds to near-random switching (the sigmoid decision becomes ~0.5 for all reward signals). These are the two most extreme non-human-like values for their respective parameters according to the winner analysis.

---

### Step 9: Compositionality Probing

We use the **Probe method** from `compositionality_analysis.md`: freeze all recurrent, input, and output weights; add one new rule vector per held-out task; optimize only that vector and measure how quickly/accurately the frozen dynamics solve the task. Fast convergence at held-out tasks = compositional generalization (the dynamics already contain the required primitives).

#### 9A: Quick Probe (Priority — 6 tasks)

Minimum viable experiment. Tests the three main transport directions at matched complexity levels.

| Type | Task | Primitives needed | Compositional demand |
|---|---|---|---|
| Control (training baseline) | `poli.antidm1` | Anti | 2-op (seen in Decision) |
| Control | `poli.dlyantigo` | Delay + Anti | 3-op (seen in Reach) |
| Control | `poli.dlyantictxgo` | Delay + Anti + Ctx | 4-op (seen in Reach) |
| **Target** | `poli.ctxdm1` | Ctx → Decision | Transport L1 |
| **Target** | `poli.antidlyms` | Anti → Match | Transport L1 |
| **Target** | `poli.antictxdlyms` | Anti + Ctx → Match | Double transport L2 |

The **generalization cost** for each model is:
$$\Delta_k = \text{steps to 80\% accuracy on held-out}_k - \text{median steps to 80\% on training controls}$$
$\Delta \approx 0$ = perfect compositional reuse. $\Delta \gg 0$ = the dynamics had to be rebuilt from scratch.

#### 9B: Full Compositional Probe (Comprehensive — 14 held-out tasks)

Run the complete held-out set from `compositionality_analysis.md` to resolve the full compositional landscape:

**Transport Context → Decision (6 tasks):** `ctxdm1`, `ctxdm2`, `ctxdlydm1`, `ctxdlydm2`, `antictxdm1`, `antictxdlydm1`

**Transport Anti → Match (3 tasks):** `antidlyms`, `antidlynms`, `anticatdlyms`

**Double Transport → Match (2 tasks):** `antictxdlyms`, `antictxcatdlyms`

Also include the 3 control tasks above. **Total per model: 17 probe tasks.**

Per-model per-task metrics:
- **Steps to 80% accuracy** (primary: measures generalization speed)
- **Asymptotic probe accuracy** (final-100-steps mean: measures generalization ceiling)
- **$\Delta$** (generalization cost relative to matched training controls)

#### 9C: Cluster Execution

```bash
# Step 1: run probes for H-50
python analysis/compare_experiments_forage.py \
    --sweep_dirs /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --mode targeted \
    --target-runs <H-50 run IDs> \
    --compute-gen \
    --gen-tasks poli.ctxdm1 poli.antidlyms poli.antictxdlyms \
                poli.antidm1 poli.dlyantigo poli.dlyantictxgo \
    --probe-steps 5000 \
    --execution-mode cluster

# Step 2: repeat with the three control cohorts (C-temp, C-reg, C-both)
# (substitute run IDs accordingly)

# Step 3: collect results locally and run aggregation
python analysis/compare_experiments_forage.py \
    --sweep_dirs /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6 \
    --mode targeted --target-runs <all run IDs> \
    --execution-mode local   # plots from cached results
```

---

### Step 10: Statistical Analysis of Compositionality Results

#### 10A: Causal Decomposition Table

For each of the 6 Quick Probe tasks, compare the four cohorts on $\Delta$ (generalization cost):

| Comparison | What it tests |
|---|---|
| H-50 vs C-temp | Role of deterministic switching (temp) alone |
| H-50 vs C-reg | Role of regularization/synaptic pruning alone |
| H-50 vs C-both | Combined effect; baseline of maximum disruption |
| C-temp vs C-both | Marginal effect of reg when temp is already broken |
| C-reg vs C-both | Marginal effect of temp when reg is already broken |

Statistics: Kruskal-Wallis (omnibus), Mann-Whitney U (pairwise), Bonferroni correction across 6 tasks.

#### 10B: Continuous Human-Similarity → Compositionality Correlation

For all probed models (H-50 + 3 × ~50 controls = ~200 models), correlate the human-similarity distances with compositionality scores. This tests whether the relationship is **continuous** (not just a threshold at the top 50):

- `grand_distance` vs mean $\Delta$ across held-out tasks (Spearman)
- Decompose: does `norm_dist_synaptic`, `norm_dist_network`, or `norm_dist_cognitive` best predict compositionality? (partial correlations)
- Key hypothesis: if synaptic developmental similarity is the mechanistic driver, `norm_dist_synaptic` should dominate the partial correlation even after controlling for the others.

#### 10C: Compositional Depth Scaling

Does the H-50 advantage grow with transport complexity?

- L1 transport (single-primitive transfer): `ctxdm1`, `antidlyms`
- L2 transport (double-primitive transfer): `antictxdlyms`

Plot mean $\Delta$ by cohort × complexity level. Hypothesis: H-50 shows **sub-linear degradation** at L2 (compositions are nearly free), while controls show **super-linear degradation** (each extra transport compounds the failure).

#### 10D: Visualizations

1. **Violin / raincloud plot**: $\Delta$ by cohort (H-50, C-temp, C-reg, C-both), faceted by task (training control vs held-out), colored by transport type.
2. **Scatter**: `grand_distance` vs compositionality score (mean asymptotic accuracy on held-out tasks), all ~200 probed models. Color = cohort, shape = transport type.
3. **Heatmap**: Model × held-out task grid, values = probe accuracy. Rows sorted by `grand_distance`. Expected pattern: top rows (H-50) should show a bright band across all held-out tasks; control cohorts should show progressive darkening.
4. **Bar chart**: Sub-distance partial correlations (synaptic / network / cognitive vs compositionality), with 95% bootstrap CIs.

---

### Expected Outcomes and Interpretation

| Pattern | Interpretation |
|---|---|
| H-50 ≫ all controls | Both temp and reg jointly drive compositionality; human-like models are compositionally superior |
| H-50 ≈ C-reg ≫ C-temp, C-both | Deterministic switching (low temp) alone drives compositionality; synaptic pruning is incidental |
| H-50 ≈ C-temp ≫ C-reg, C-both | Regularization / developmental pruning alone drives compositionality; exploration dynamics are incidental |
| All cohorts similar | Human-likeness and compositionality are dissociated; the features that make models brain-like do not predict compositional generalization |

The strongest scientific claim — that **brain-like development causally enables compositional generalization** — requires the first row. The partial-correlation analysis (10B) would then further tell us *which* of the three biological comparison axes (synaptic development, network topology, cognitive foraging) is the mechanistic link.

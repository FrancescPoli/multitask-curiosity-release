# Future Steps: Breaking the "Bag of Bits"

To ensure our models learn **Rich, Compositional Dynamics** rather than disjoint "Autapse" solutions (as described by Khona et al., 2023), we propose the following interventions.

## 1. The Anti-Orthogonality Hack (Correlated Rules)

**The Problem:**
Currently, our Rule Inputs are **One-Hot Vectors** (Orthogonal).
*   Rule A: `[1, 0, 0]`
*   Rule B: `[0, 1, 0]`
This allows the network to assign Neuron 1 to Task A and Neuron 2 to Task B with zero interference (a "Bag of Bits").

**The Hack: Low-Rank Random Embeddings (The "Crowded Room" Approach)**

To force interference without leaking task structure (e.g., Reach clustering together), we use **Random Projection**.

1.  **Low-Dimensional Bottleneck**: Generate 20 task vectors in a **4-dimensional subspace** ($z \in \mathbb{R}^4$).
2.  **Random Projection**: Map these to the 20D input layer ($v = W_{proj} z$).
    *   $W_{proj}$ is a fixed random orthogonal matrix.

**Why it works:**
1.  **Guaranteed Interference**: In 4D, 20 vectors *must* overlap significantly (expected $|\cos \theta| \approx 0.5$).
2.  **Zero Information Leak**: The overlap is random. `go` (Reach) might be highly correlated with `multidm` (Decision).
3.  **Forced Separation**: The network receives a messy, ambiguous input. To perform `go`, it *must* use recurrent dynamics (lateral inhibition) to suppress the interfering `multidm` signal. Autapses fail here.

**Biological Plausibility:**
This is **more plausible** than One-Hot encoding.
*   Neural representations in the brain (e.g., PFC, PPC) are rarely orthogonal.
*   They naturally exhibit **Mixed Selectivity** and coarse coding.
*   Input signals are high-dimensional and correlated. The brain *relies* on recurrent dynamics to separate these overlapping patterns (Pattern Separation).

---

## 2. The Small Network Constraint ($N=48$)

**The Question:** Why is $N=48$ sufficient to force structure for 20 tasks?

**Capacity Analysis:**
*   **"Bag of Bits" Cost**: To solve a task independently, you need at least 2 distinct states (e.g., Saccade Left / Saccade Right).
*   **Minimum Neurons**: If each task uses disjoint neurons, 20 tasks $\requires$ 40 neurons. (2 neurons $\times$ 20 tasks).
*   **The Squeeze**:
    *   If $N=48$, we have almost zero "slack". The network is at the **Capacity Limit** for a disjoint solution.
    *   Any inefficient allocation (e.g., using 3 neurons for a robust independent task) breaks the system.
    *   **Mixed Selectivity**: To fit 20 tasks into 48 neurons *robustly*, the network is forced to **Result**: Reuse neurons. Neuron 7 might encode "Delay" for *both* Task A and Task B.
    *   This sharing of computational resources is the definition of **Compositionality**.

---

## 3. Targeted Regularization: The Diagonal Penalty

If L1 (Sparsity) accidentally favors autapses (1 weight vs 2 weights), we need a scalpel, not a hammer.

**Proposal: Diagonal L2 Penalty**
$$ L_{diag} = \lambda \sum_{i} (W_{rec}[i,i])^2 $$

**Mechanism:**
*   Explicitly penalizes the **Self-Loop (Autapse)** weights on the diagonal.
*   **Does NOT** penalize off-diagonal weights ($W_{ij}$).
*   **Effect**: It makes autapses "expensive". The network prefers to build loops via neighbors ($i \to j \to i$), promoting local recurrence and mixing.

**Combination**:
*   **Diagonal Penalty** (Kill Autapses) + **Distance Penalty** (Localize Wiring).
*   Result: Local Micro-Circuits ($i \leftrightarrow j$) rather than isolated dots.

---

## 4. Regularization Scope Asymmetry (L1 vs. Distance)

**The Issue:**
The two regularization regimes in our final sweep (V8) differ not only in cost function but also in *scope*:
*   **L1 regime:** $\ell_1$ penalty applied to all trainable weight matrices — input ($W_{\text{in}}$), recurrent ($W_{\text{rec}}$), and readout ($W_{\text{out}}$) — biases excluded.
*   **Distance regime:** Distance-weighted $\ell_1$ penalty applied **only** to the recurrent weights. The input weights (including the $100$-d rule projection) and the readout weights are completely unregularized: no L1, no proximal L1, no Adam weight decay.

**Why this matters:**
1.  **Compensation risk.** With $W_{\text{rec}}$ heavily penalized but $W_{\text{in}}$ and $W_{\text{out}}$ free, the optimizer may offload computation onto the input/readout layers rather than reorganizing the recurrent connectivity. This would mask the intended effect of the distance penalty on the lateral wiring.
2.  **Confounded comparisons.** Differences between the L1 and Distance branches conflate (a) the type of cost (uniform $\ell_1$ vs. distance-weighted $\ell_1$) with (b) the scope (all weights vs. recurrent only). A reviewer can argue that any branch effect is due to scope rather than to the wiring-cost hypothesis itself.

**Actions:**

1.  **Diagnostic check.** Compute weight norms (e.g., Frobenius, or per-row $\ell_1$) of $W_{\text{in}}$, $W_{\text{rec}}$, and $W_{\text{out}}$ for every trained model in both branches and ask:
    *   Are the unregularized layers ($W_{\text{in}}$, $W_{\text{out}}$) in the Distance branch noticeably larger than the corresponding layers in the L1 branch?
    *   Are they noticeably larger than at initialization?
    
    If yes → weights are inflating to compensate; if no → the asymmetry is benign and we can report it as such.

2.  **Control sweep** to disentangle cost-type from scope. Run *one* of the following alongside the existing branches:
    *   **Option A — match Distance's scope:** L1 restricted to the recurrent matrix only (`l1_on = 'recurrent-only'`, already supported in `run.py`).
    *   **Option B — match L1's scope:** Distance penalty on $W_{\text{rec}}$ *plus* L1 on $W_{\text{in}}$ and $W_{\text{out}}$.
    
    Either option isolates the effect of the cost function from the effect of the scope.

---

## 4.1 Implementation plan for the diagnostic check (§4 point 1)

**Goal.** Determine whether the Distance branch inflates $W_{\text{in}}$ / $W_{\text{out}}$ relative to the L1 branch, so we know whether the V8 contrast is confounded by regularization *scope*.

**Sweep.** `Z:/fp02/logs/sweep/forage_v5.1/forage_v6/` — all 3 136 runs (V6 + V7 + V8 are folded into this directory via `--base-dir`).

**Metric.** Match the existing W_rec pipeline ([analysis/extract_weights_evolution.py](analysis/extract_weights_evolution.py)): per matrix, compute
- `l1_sum` — total $\ell_1$ norm (sum of $|w|$)
- `l0_norm` — count of $|w| > 10^{-5}$

Compute on the final state (`state_dict.pt`) for $W_{\text{in}}$, $W_{\text{rec}}$, $W_{\text{out}}$. Reference initialization values are computed once from any single `state_step000000.pt` (deterministic given fixed `seed=42`).

**Step 1 — Extractor** `analysis/Regularization_scope/extract_io_weight_norms.py` (new, no edits to existing code)
- Reuses `load_meta`, `build_model_from_meta`, `load_state_into_model` from [curiosity/utils/model_loader.py](curiosity/utils/model_loader.py); defines local `get_Win` / `get_Wrec` / `get_Wout` helpers mirroring [analysis/plotting/utils.py:58-76](analysis/plotting/utils.py#L58-L76).
- Output CSV: `analysis/Regularization_scope/io_weight_norms.csv` with columns
    `run_id, w_in_l1_sum, w_in_l0, w_rec_l1_sum, w_rec_l0, w_out_l1_sum, w_out_l0`
- Resume-from-CSV (mirrors [analysis/Synaptic_analysis/extract_weights.py](analysis/Synaptic_analysis/extract_weights.py)). Run locally first; move to cluster only if too slow.

**Step 2 — Analysis** `analysis/Regularization_scope/compare_branches.py` (new)
Merge `io_weight_norms.csv` with [analysis/grand_unified_metrics_v2.csv](analysis/grand_unified_metrics_v2.csv) on `run_id`. All branch / reg / foraging / accuracy metadata is already there.
1. Per-branch ratio `l1_sum_final / l1_sum_init` for $W_{\text{in}}$ and $W_{\text{out}}$ — inflation vs init.
2. Matched-config L1-vs-Distance contrast of final $W_{\text{in}}$ and $W_{\text{out}}$ L1 sums (group on beta/alpha/travel/temp/eps, paired Wilcoxon).
3. Sanity: $W_{\text{rec}}$ L1 sum should be smaller in Distance branch (regularization is biting where expected).
4. Bonus: do $W_{\text{in}}$ / $W_{\text{out}}$ norms correlate with `fraction_solved` or `dist_synaptic` within each branch?

**Decision rule.** If (1) shows ratios ≈ 1 and (2) shows no L1-vs-Distance gap, scope asymmetry is benign — note in SI. If clear inflation, schedule the V8 control sweep (Option A: `l1_on='recurrent-only'`).

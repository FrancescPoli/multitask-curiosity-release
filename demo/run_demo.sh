#!/usr/bin/env bash
# =============================================================================
# Demo: train a small curious RNN, analyse its connectome, and reproduce the
# main results from the shipped metric tables.
#
# Run from the repository root:
#     bash demo/run_demo.sh
#
# Interpreters can be overridden, e.g.
#     PYTHON=/path/to/python RSCRIPT="/path/to/Rscript" bash demo/run_demo.sh
#
# Expected total run time on a normal desktop (CPU only): ~5 minutes.
# New files are written to demo/output/; step 3 regenerates the publication
# figures in analysis/Final_analyses_and_plots/Figures/ (identical to the
# shipped versions).
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
OUT="demo/output"
SWEEP="$OUT/sweep"
mkdir -p "$SWEEP"

# -----------------------------------------------------------------------------
# 1. Train one RNN with the MVT (curiosity-driven) curriculum.
#    Same settings as the paper's sweep (experiments/run_sweep_forage.py,
#    config forage_v6), but 500 instead of 600,000 training steps.
# -----------------------------------------------------------------------------
echo "=== [1/3] Training a demo RNN with the MVT curriculum (500 steps) ==="
"$PYTHON" run.py \
    --logdir "$SWEEP" --run-name run_demo \
    --steps 500 --batch-size 16 --log-every 100 --save-every 100 --seed 42 \
    --hidden 100 --activation relu --w-rec-init randortho --sigma-x 0.01 \
    --grad-clip-mode value \
    --tasks poli.go poli.rtgo poli.dlygo poli.antigo poli.ctxgo \
            poli.dlyantigo poli.dlyctxgo poli.antictxgo poli.dlyantictxgo \
            poli.rtantictxgo poli.dm1 poli.dm2 poli.dlydm1 poli.multidm \
            poli.antidm1 poli.antidlydm1 poli.dlyms poli.dlynms \
            poli.catdlyms poli.catdlynms \
    --rule-encoding lowrank --rule-dim-low 4 --rule-dim-out 100 \
    --distance-penalty --distance-path brain/assets/euclidean_distances_schaefer100.npy \
    --distance-weight 1e-4 --distance-power 2.0 \
    --foraging mvt --forage-alpha-local 0.03 --forage-beta-global 0.001 \
    --forage-travel-steps 50 --forage-min-block-steps 1 \
    --forage-temperature 0.01 --forage-eps 0.0

# -----------------------------------------------------------------------------
# 2. Analyse the trained network: weight trajectories (synaptic density) and
#    graph topology (modularity, efficiency, rich club) of the final W_rec.
# -----------------------------------------------------------------------------
echo "=== [2/3] Extracting weights and computing network topology ==="
"$PYTHON" analysis/Synaptic_analysis/extract_weights.py \
    --sweep_dir "$SWEEP" --output "$OUT/population_weights_demo.csv" --save_npy
"$PYTHON" analysis/Network_analysis/compute_rnn_topology.py \
    --sweep_dir "$SWEEP" --output "$OUT/rnn_topology_demo.csv"

# -----------------------------------------------------------------------------
# 3. Reproduce the main statistics and figures from the shipped metric tables
#    (3,136 trained networks + HCP-YA reference).
# -----------------------------------------------------------------------------
echo "=== [3/3] Reproducing main results (network, synaptic, compositional) ==="
"$RSCRIPT" analysis/Final_analyses_and_plots/pub_topology_consolidated.R     | tee "$OUT/pub_topology.log"
"$RSCRIPT" analysis/Final_analyses_and_plots/pub_synaptic_consolidated.R     | tee "$OUT/pub_synaptic.log"
"$RSCRIPT" analysis/compositional_analysis/pub_compositional_generalisation.R | tee "$OUT/pub_compositional.log"

echo
echo "Demo complete. Outputs in $OUT/ and analysis/Final_analyses_and_plots/Figures/"

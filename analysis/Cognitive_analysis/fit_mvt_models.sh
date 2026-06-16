#!/bin/bash
#SBATCH -J fit_mvt
#SBATCH -o /imaging/astle/fp02/RNN/logs/slurm/fit_mvt_%A_%a.out
#SBATCH -e /imaging/astle/fp02/RNN/logs/slurm/fit_mvt_%A_%a.err
#SBATCH -p Main
#SBATCH -q normal
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=16G
#SBATCH --exclude=node-k01,node-k12
#SBATCH --array=0-7

# ----------------------------------------------------------------------
# Fit MVT GLM to all model runs in a sweep directory, split across 8
# parallel array tasks. Each task handles run_dirs[shard::N_SHARDS]
# and writes to its own CSV (model_mvt_params_fitted_shard{0..7}of8.csv).
#
# Submit from the project root with:
#   conda activate ngym39                                    # required
#   sbatch analysis/Cognitive_analysis/fit_mvt_models.sh [SWEEP_DIR] [OUT_CSV]
#
# After all 8 array tasks finish, merge the shards into a single CSV:
#   python analysis/Cognitive_analysis/fit_mvt_models.py \
#       --sweep_dir <anything> --merge \
#       --output analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv
#
# Defaults if args omitted:
#   SWEEP_DIR = /imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6
#   OUT_CSV   = analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv
#
# NOTE: the SBATCH -o / -e paths ABOVE are absolute on purpose. SLURM
# evaluates them before the script body runs, so relative paths fail
# silently if the cwd at submission time isn't the project root.
# To change shard count, update --array=0-N and N_SHARDS together.
# ----------------------------------------------------------------------

N_SHARDS=8

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
eval "$(conda shell.bash hook)"
conda activate ngym39

if [ -f "/imaging/astle/fp02/RNN/analysis/Cognitive_analysis/fit_mvt_models.py" ]; then cd "/imaging/astle/fp02/RNN";
elif [ -d "multitask-curiosity" ]; then cd multitask-curiosity; fi

SWEEP_DIR="${1:-/imaging/astle/fp02/logs/sweep/forage_v5.1/forage_v6}"
OUT_CSV="${2:-analysis/Cognitive_analysis/Results/model_mvt_params_fitted.csv}"

mkdir -p "$(dirname "${OUT_CSV}")"

echo "Node:    $(hostname)"
echo "Pwd:     $(pwd)"
echo "Start:   $(date)"
echo "Sweep:   ${SWEEP_DIR}"
echo "Output:  ${OUT_CSV}"
echo "Shard:   ${SLURM_ARRAY_TASK_ID}/${N_SHARDS}"
echo "Python:  $(which python)"

python -u analysis/Cognitive_analysis/fit_mvt_models.py \
    --sweep_dir "${SWEEP_DIR}" \
    --output "${OUT_CSV}" \
    --n_shards "${N_SHARDS}" \
    --shard "${SLURM_ARRAY_TASK_ID}"

echo "End:     $(date)"

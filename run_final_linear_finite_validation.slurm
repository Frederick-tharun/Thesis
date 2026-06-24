#!/bin/bash
#SBATCH --job-name=lf_ft_final
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/lf_ft_final_%j.out
#SBATCH --error=logs/lf_ft_final_%j.err

set -euo pipefail
cd ~/Thesis

if [[ -x .conda_env/bin/python ]]; then
    PY=.conda_env/bin/python
elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    PY=python3
fi

RESULT_ROOT="selected_controller_results/final_linear_finite_${SLURM_JOB_ID}"
mkdir -p "$RESULT_ROOT"

$PY -m py_compile main.py model.py neuron_controllers.py \
    control_experiment.py plotting.py config.py data_loader.py optimize_model.py
$PY -m unittest tests/test_dummy_optimizer_compat.py -v

LINEAR_ROOT="$RESULT_ROOT/linear_feedback_K1p0"
LINEAR_SRC="outputs/chaotic_bursting/control/linear_feedback"
LINEAR_MARKER="${LINEAR_ROOT}.started"
touch "$LINEAR_MARKER"

$PY main.py \
    --dataset hr \
    --hr-mode chaotic_bursting \
    --optimizer dummy \
    --control \
    --controller linear_feedback \
    --control-k 1.0

test -s "$LINEAR_SRC/control_summary.json"
test "$LINEAR_SRC/control_summary.json" -nt "$LINEAR_MARKER"
mkdir -p "$LINEAR_ROOT"
cp -a "$LINEAR_SRC"/. "$LINEAR_ROOT"/
cp outputs/chaotic_bursting/best_params.json "$LINEAR_ROOT/esn_best_params.json"
cp outputs/chaotic_bursting/metrics.json "$LINEAR_ROOT/prediction_metrics.json"

FINITE_ROOT="$RESULT_ROOT/finite_time_K0p4582142857142857_s0p8"
FINITE_SRC="outputs/chaotic_bursting/control/finite_time"
FINITE_MARKER="${FINITE_ROOT}.started"
touch "$FINITE_MARKER"

$PY main.py \
    --dataset hr \
    --hr-mode chaotic_bursting \
    --optimizer dummy \
    --control \
    --controller finite_time \
    --control-k 0.4582142857142857 \
    --finite-s 0.8

test -s "$FINITE_SRC/control_summary.json"
test "$FINITE_SRC/control_summary.json" -nt "$FINITE_MARKER"
mkdir -p "$FINITE_ROOT"
cp -a "$FINITE_SRC"/. "$FINITE_ROOT"/
cp outputs/chaotic_bursting/best_params.json "$FINITE_ROOT/esn_best_params.json"
cp outputs/chaotic_bursting/metrics.json "$FINITE_ROOT/prediction_metrics.json"

cp "$0" "$RESULT_ROOT/"
git rev-parse HEAD > "$RESULT_ROOT/GIT_COMMIT.txt"

echo "FINAL LINEAR + FINITE-TIME VALIDATION COMPLETED"
echo "Results: $RESULT_ROOT"

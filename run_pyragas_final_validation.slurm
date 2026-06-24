#!/bin/bash
#SBATCH --job-name=pyr_final_val
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/pyr_final_val_%j.out
#SBATCH --error=logs/pyr_final_val_%j.err

set -euo pipefail
cd ~/Thesis

if [[ -x .conda_env/bin/python ]]; then
    PY=.conda_env/bin/python
elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    PY=python3
fi

ROOT="selected_pyragas_results/final_K0p8_delay2400_sign_-1_${SLURM_JOB_ID}"
SRC="outputs/chaotic_bursting/control/pyragas"
MARKER="${ROOT}.started"

mkdir -p selected_pyragas_results
touch "$MARKER"

$PY -m py_compile main.py model.py neuron_controllers.py \
    control_experiment.py plotting.py config.py data_loader.py

$PY main.py \
    --dataset hr \
    --hr-mode chaotic_bursting \
    --optimizer dummy \
    --control \
    --controller pyragas \
    --control-k 0.8 \
    --pyragas-delay 2400 \
    --pyragas-sign -1

SUMMARY="$SRC/control_summary.json"

test -s "$SUMMARY"
test "$SUMMARY" -nt "$MARKER"

mkdir -p "$ROOT"
cp -a "$SRC"/. "$ROOT"/
cp outputs/chaotic_bursting/best_params.json "$ROOT/esn_best_params.json"
cp outputs/chaotic_bursting/metrics.json "$ROOT/prediction_metrics.json"

echo "FINAL PYRAGAS VALIDATION COMPLETED"
echo "Results: $ROOT"

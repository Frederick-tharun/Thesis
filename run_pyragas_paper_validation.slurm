#!/bin/bash
#SBATCH --job-name=pyr_paper_val
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/pyr_paper_val_%j.out
#SBATCH --error=logs/pyr_paper_val_%j.err

set -uo pipefail

cd ~/Thesis
mkdir -p logs selected_pyragas_results

if [ -x ".conda_env/bin/python" ]; then
    PY=".conda_env/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

echo "Python: $PY"
$PY --version

echo "Checking updated source files..."
$PY -m py_compile \
    main.py \
    model.py \
    neuron_controllers.py \
    control_experiment.py \
    plotting.py \
    config.py \
    data_loader.py || exit 1

echo "Running known invalid Pyragas candidate"
echo "K=0.1, delay=320, sign=-1"

$PY main.py \
    --dataset hr \
    --hr-mode chaotic_bursting \
    --optimizer dummy \
    --control \
    --controller pyragas \
    --control-k 0.1 \
    --pyragas-delay 320 \
    --pyragas-sign -1

STATUS=$?

if [ "$STATUS" -eq 0 ]; then
    SRC="outputs/chaotic_bursting/control/pyragas"
    DEST="selected_pyragas_results/paper_validation_delay320_${SLURM_JOB_ID}"

    mkdir -p "$DEST"
    cp -a "$SRC"/. "$DEST"/

    echo
    echo "===== PAPER-ALIGNED PYRAGAS VALIDATION ====="

    jq '.best | {
      K,
      pyragas_delay,
      pyragas_sign,
      pyragas_selection_status,
      pyragas_quality_pass,
      pyragas_quality_issues,
      pyragas_rhythm_type,
      pyragas_detected_peak_count,
      pyragas_detected_cycle_count,
      pyragas_cycle_counts_by_window,
      pyragas_cycle_window_coverage,
      pyragas_rhythm_interval_cv,
      pyragas_empirical_period_steps,
      pyragas_empirical_period_time,
      pyragas_empirical_recurrence_error_norm,
      pyragas_empirical_recurrence_correlation,
      pyragas_empirical_tail_closure_error_norm,
      pyragas_x_amplitude_ratio,
      pyragas_drift_ratio,
      pyragas_tail_activity_ratio
    }' "$SRC/control_summary.json"

    echo
    echo "Saved result: $DEST"
fi

exit "$STATUS"

#!/bin/bash
#SBATCH --job-name=pyr_final_search
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/pyr_final_search_%j.out
#SBATCH --error=logs/pyr_final_search_%j.err

set -uo pipefail
cd ~/Thesis

if [ -x ".conda_env/bin/python" ]; then
    PY=".conda_env/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

$PY -m py_compile \
    main.py model.py neuron_controllers.py \
    control_experiment.py plotting.py config.py data_loader.py || exit 1

DELAYS=(1 2 3 5 10 20 40 80 160 320 640 1280 1600 2400 3200)
ROOT="selected_pyragas_results/paper_joint_search_${SLURM_JOB_ID}"
SRC="outputs/chaotic_bursting/control/pyragas"

mkdir -p "$ROOT"

for DELAY in "${DELAYS[@]}"; do
    echo
    echo "============================================================"
    echo "Delay=$DELAY, sign=-1, K=0.02–0.80"
    echo "Started: $(date)"
    echo "============================================================"

    $PY main.py \
      --dataset hr \
      --hr-mode chaotic_bursting \
      --optimizer dummy \
      --control \
      --controller pyragas \
      --auto-control-k \
      --pyragas-delay "$DELAY" \
      --pyragas-sign -1 \
      --k-min 0.02 \
      --k-max 0.80 \
      --k-num 31 \
      --k-refine-num 21

    STATUS=$?

    if [ "$STATUS" -eq 0 ] && [ -f "$SRC/control_summary.json" ]; then
        DEST="$ROOT/delay_${DELAY}_sign_-1"
        mkdir -p "$DEST"
        cp -a "$SRC"/. "$DEST"/

        jq '.best | {
          K,
          pyragas_delay,
          pyragas_quality_pass,
          pyragas_rhythm_type,
          pyragas_detected_cycle_count,
          pyragas_rhythm_interval_cv,
          pyragas_empirical_period_steps,
          pyragas_empirical_recurrence_error_norm,
          pyragas_empirical_recurrence_correlation,
          pyragas_quality_issues
        }' "$DEST/control_summary.json"
    else
        echo "FAILED delay=$DELAY status=$STATUS"
    fi
done

$PY - "$ROOT" <<'PY'
import csv
import glob
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []

for path in glob.glob(str(root / "delay_*" / "control_summary.json")):
    with open(path) as f:
        best = json.load(f)["best"]

    score = best.get("selection_score")
    if score is None or not math.isfinite(float(score)):
        score = float("inf")

    rows.append({
        "folder": str(Path(path).parent),
        "delay": best.get("pyragas_delay"),
        "K": best.get("K"),
        "quality_pass": bool(best.get("pyragas_quality_pass", False)),
        "rhythm_type": best.get("pyragas_rhythm_type"),
        "cycle_count": best.get("pyragas_detected_cycle_count"),
        "rhythm_cv": best.get("pyragas_rhythm_interval_cv"),
        "period_steps": best.get("pyragas_empirical_period_steps"),
        "recurrence_error": best.get(
            "pyragas_empirical_recurrence_error_norm"
        ),
        "recurrence_correlation": best.get(
            "pyragas_empirical_recurrence_correlation"
        ),
        "amplitude_ratio": best.get("pyragas_x_amplitude_ratio"),
        "selection_score": score,
        "issues": best.get("pyragas_quality_issues"),
    })

rows.sort(key=lambda r: (not r["quality_pass"], r["selection_score"]))

with open(root / "joint_search_ranking.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

passing = [r for r in rows if r["quality_pass"]]

print("\n===== JOINT SEARCH RANKING =====")
for row in rows:
    print(row)

if passing:
    winner = passing[0]
    (root / "best_candidate_path.txt").write_text(winner["folder"])
    print("\n===== QUALITY-PASSING WINNER =====")
    print(winner)
else:
    (root / "NO_QUALITY_PASS.txt").write_text(
        "No tested delay/K combination passed all Pyragas checks.\n"
    )
    print("\nNO QUALITY-PASSING CANDIDATE")
PY

if [ -s "$ROOT/best_candidate_path.txt" ]; then
    WINNER="$(cat "$ROOT/best_candidate_path.txt")"
    mkdir -p "$ROOT/FINAL_CANDIDATE"
    cp -a "$WINNER"/. "$ROOT/FINAL_CANDIDATE/"
    echo "Final candidate copied to: $ROOT/FINAL_CANDIDATE"
else
    echo "No final candidate was created."
fi

echo "Finished: $(date)"
echo "All results: $ROOT"

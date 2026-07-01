#!/bin/bash
#SBATCH --job-name=final_thesis_pipeline
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/final_thesis_pipeline_%j.out
#SBATCH --error=logs/final_thesis_pipeline_%j.err

set -euo pipefail

mkdir -p logs

module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/thesis_final"

export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"

JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
PROJECT_DIR="$PWD"
FINAL_ROOT="$HOME/Thesis_evidence_20260624/final_thesis_run_${JOB_ID}"

mkdir -p "$FINAL_ROOT"/{00_manifest,01_prediction_all_regimes,02_bo_optimization,03_linear_feedback,04_finite_time,05_pyragas,06_comparison_tables,07_report_figures,08_logs}

echo "========================================================================"
echo "FINAL THESIS PIPELINE"
echo "========================================================================"
echo "Project dir : $PROJECT_DIR"
echo "Final root  : $FINAL_ROOT"
echo "Job ID      : $JOB_ID"
echo "Date        : $(date)"
echo "Git commit  : $(git rev-parse HEAD)"
echo "Git branch  : $(git branch --show-current)"
echo "========================================================================"

git rev-parse HEAD > "$FINAL_ROOT/00_manifest/GIT_COMMIT.txt"
git branch --show-current > "$FINAL_ROOT/00_manifest/GIT_BRANCH.txt"
date > "$FINAL_ROOT/00_manifest/RUN_DATE.txt"
python --version > "$FINAL_ROOT/00_manifest/PYTHON_VERSION.txt"
python -m pip freeze > "$FINAL_ROOT/00_manifest/requirements_freeze.txt"

echo "========================================================================"
echo "STEP 1: ESN DIGITAL TWIN PREDICTION FOR ALL HR REGIMES"
echo "========================================================================"

rm -rf outputs

python main.py \
  --dataset hr \
  --run-all-regimes \
  --optimizer auto \
  --clean-output

cp -a outputs "$FINAL_ROOT/01_prediction_all_regimes/outputs_after_prediction"

# Copy BO/prediction files to easy report folders if they exist
find outputs -type f \( \
  -name "optimizer_convergence.png" -o \
  -name "bo_objective_landscape_best_optimizer.png" -o \
  -name "optimizer_history.csv" -o \
  -name "optimizer_ranking_table.csv" -o \
  -name "best_params.json" -o \
  -name "metrics.json" -o \
  -name "results_all_states.png" -o \
  -name "results_full_zoom.png" -o \
  -name "results_zoom_comparison.png" -o \
  -name "spike_event_comparison.png" \
\) -exec cp --parents {} "$FINAL_ROOT/02_bo_optimization/" \;

echo "========================================================================"
echo "STEP 2: LINEAR FEEDBACK CONTROL ON CHAOTIC BURSTING"
echo "========================================================================"

rm -rf outputs/chaotic_bursting

python main.py \
  --dataset hr \
  --hr-mode chaotic_bursting \
  --optimizer dummy \
  --clean-output \
  --control \
  --controller linear_feedback \
  --auto-control-k \
  --k-min 0.01 \
  --k-max 2.0 \
  --k-num 25 \
  --k-refine-num 20 \
  --control-start-frac 0.2 \
  --control-target-mode rest_state

mkdir -p "$FINAL_ROOT/03_linear_feedback"
cp -a outputs/chaotic_bursting "$FINAL_ROOT/03_linear_feedback/final_linear_feedback"

echo "========================================================================"
echo "STEP 3: FINITE-TIME CONTROL S AND K SWEEP"
echo "========================================================================"

mkdir -p "$FINAL_ROOT/04_finite_time/s_sweep"

for S in 0.3 0.5 0.7 0.8 0.9; do
  LABEL="s_${S//./p}"
  echo "Running finite-time sweep for s=$S"

  rm -rf outputs/chaotic_bursting

  python main.py \
    --dataset hr \
    --hr-mode chaotic_bursting \
    --optimizer dummy \
    --clean-output \
    --control \
    --controller finite_time \
    --finite-s "$S" \
    --auto-control-k \
    --k-min 0.01 \
    --k-max 1.5 \
    --k-num 25 \
    --k-refine-num 20 \
    --control-start-frac 0.2 \
    --control-target-mode rest_state

  cp -a outputs/chaotic_bursting "$FINAL_ROOT/04_finite_time/s_sweep/$LABEL"
done

echo "========================================================================"
echo "STEP 4: FINAL SELECTED FINITE-TIME CONTROL"
echo "========================================================================"

rm -rf outputs/chaotic_bursting

python main.py \
  --dataset hr \
  --hr-mode chaotic_bursting \
  --optimizer dummy \
  --clean-output \
  --control \
  --controller finite_time \
  --finite-s 0.8 \
  --control-k 0.4582142857142857 \
  --control-start-frac 0.2 \
  --control-target-mode rest_state

cp -a outputs/chaotic_bursting "$FINAL_ROOT/04_finite_time/final_finite_time_K0p458214_s0p8"

echo "========================================================================"
echo "STEP 5: PYRAGAS DELAY / SIGN / K SWEEP"
echo "========================================================================"

mkdir -p "$FINAL_ROOT/05_pyragas/delay_sign_k_sweep"

for SIGN in -1 1; do
  for DELAY in 20 80 320 640 1280 1600 2400 3200; do
    SIGN_LABEL="sign_${SIGN}"
    SIGN_LABEL="${SIGN_LABEL/-/minus}"
    LABEL="delay_${DELAY}_${SIGN_LABEL}"

    echo "Running Pyragas sweep for delay=$DELAY sign=$SIGN"

    rm -rf outputs/chaotic_bursting

    python main.py \
      --dataset hr \
      --hr-mode chaotic_bursting \
      --optimizer dummy \
      --clean-output \
      --control \
      --controller pyragas \
      --pyragas-delay "$DELAY" \
      --pyragas-sign="$SIGN" \
      --auto-control-k \
      --k-min 0.02 \
      --k-max 0.8 \
      --k-num 13 \
      --k-refine-num 12 \
      --control-start-frac 0.2 \
      --control-target-mode rest_state

    cp -a outputs/chaotic_bursting "$FINAL_ROOT/05_pyragas/delay_sign_k_sweep/$LABEL"
  done
done

echo "========================================================================"
echo "STEP 6: FINAL SELECTED PYRAGAS CONTROL"
echo "========================================================================"

rm -rf outputs/chaotic_bursting

python main.py \
  --dataset hr \
  --hr-mode chaotic_bursting \
  --optimizer dummy \
  --clean-output \
  --control \
  --controller pyragas \
  --pyragas-delay 2400 \
  --pyragas-sign=-1 \
  --control-k 0.8 \
  --control-start-frac 0.2 \
  --control-target-mode rest_state

cp -a outputs/chaotic_bursting "$FINAL_ROOT/05_pyragas/final_pyragas_K0p8_delay2400_sign_minus1"

echo "========================================================================"
echo "STEP 7: BUILD FINAL SUMMARY TABLES"
echo "========================================================================"

FINAL_ROOT="$FINAL_ROOT" python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["FINAL_ROOT"])
rows = []

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def add_control_result(label, controller, folder):
    folder = Path(folder)
    summary_paths = list(folder.rglob("control_summary.json"))
    if not summary_paths:
        rows.append({
            "label": label,
            "controller": controller,
            "status": "missing_control_summary"
        })
        return

    summary = load_json(summary_paths[0]) or {}
    pred_paths = list(folder.rglob("metrics.json"))
    pred = load_json(pred_paths[0]) if pred_paths else {}

    row = {
        "label": label,
        "controller": controller,
        "status": "ok",
        "optimizer": summary.get("optimizer", ""),
        "K": summary.get("best_k", summary.get("control_k", summary.get("K", ""))),
        "finite_s": summary.get("finite_s", ""),
        "pyragas_delay": summary.get("pyragas_delay", ""),
        "pyragas_sign": summary.get("pyragas_sign", ""),
        "stable": summary.get("stable", ""),
        "target_rmse_state": summary.get("target_rmse_state", summary.get("target_rmse", "")),
        "settling_time": summary.get("settling_time", ""),
        "control_energy": summary.get("control_energy", ""),
        "spike_count_after": summary.get("spike_count_after", ""),
        "spike_reduction_percent": summary.get("spike_reduction_percent", ""),
        "pyragas_quality_pass": summary.get("pyragas_quality_pass", ""),
        "pyragas_rhythm_type": summary.get("pyragas_rhythm_type", ""),
        "pyragas_rhythm_interval_cv": summary.get("pyragas_rhythm_interval_cv", ""),
        "pyragas_empirical_period_steps": summary.get("pyragas_empirical_period_steps", ""),
        "prediction_nrmse_x": pred.get("nrmse_x", pred.get("NRMSE x", "")) if isinstance(pred, dict) else "",
        "summary_path": str(summary_paths[0].relative_to(root)),
    }
    rows.append(row)

add_control_result(
    "final_linear_feedback",
    "linear_feedback",
    root / "03_linear_feedback/final_linear_feedback"
)

add_control_result(
    "final_finite_time_K0p458214_s0p8",
    "finite_time",
    root / "04_finite_time/final_finite_time_K0p458214_s0p8"
)

add_control_result(
    "final_pyragas_K0p8_delay2400_sign_minus1",
    "pyragas",
    root / "05_pyragas/final_pyragas_K0p8_delay2400_sign_minus1"
)

# Add finite-time sweep rows
for d in sorted((root / "04_finite_time/s_sweep").glob("*")):
    add_control_result(f"finite_sweep_{d.name}", "finite_time", d)

# Add Pyragas sweep rows
for d in sorted((root / "05_pyragas/delay_sign_k_sweep").glob("*")):
    add_control_result(f"pyragas_sweep_{d.name}", "pyragas", d)

out_dir = root / "06_comparison_tables"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "final_controller_comparison.csv"
fieldnames = sorted({k for row in rows for k in row.keys()})
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md_path = out_dir / "final_controller_comparison.md"
with open(md_path, "w") as f:
    f.write("# Final Controller Comparison\n\n")
    f.write("| label | controller | K | s | delay | sign | stable | target RMSE | energy | Pyragas pass | rhythm |\n")
    f.write("|---|---|---:|---:|---:|---:|---|---:|---:|---|---|\n")
    for r in rows:
        f.write(
            f"| {r.get('label','')} | {r.get('controller','')} | {r.get('K','')} | "
            f"{r.get('finite_s','')} | {r.get('pyragas_delay','')} | {r.get('pyragas_sign','')} | "
            f"{r.get('stable','')} | {r.get('target_rmse_state','')} | {r.get('control_energy','')} | "
            f"{r.get('pyragas_quality_pass','')} | {r.get('pyragas_rhythm_type','')} |\n"
        )

manifest = {
    "final_root": str(root),
    "final_results": {
        "linear_feedback": "03_linear_feedback/final_linear_feedback",
        "finite_time": "04_finite_time/final_finite_time_K0p458214_s0p8",
        "pyragas": "05_pyragas/final_pyragas_K0p8_delay2400_sign_minus1",
    },
    "comparison_csv": "06_comparison_tables/final_controller_comparison.csv",
    "comparison_md": "06_comparison_tables/final_controller_comparison.md",
}
with open(root / "00_manifest/final_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved {csv_path}")
print(f"Saved {md_path}")
print(f"Saved {root / '00_manifest/final_manifest.json'}")
PY

echo "========================================================================"
echo "STEP 8: COPY KEY FIGURES TO REPORT FIGURES"
echo "========================================================================"

find "$FINAL_ROOT" -type f \( \
  -name "*.png" -o \
  -name "*.csv" -o \
  -name "*.md" -o \
  -name "*.json" \
\) | sort > "$FINAL_ROOT/00_manifest/ALL_OUTPUT_FILES.txt"

find "$FINAL_ROOT" -type f \( \
  -name "results_full_zoom.png" -o \
  -name "results_all_states.png" -o \
  -name "optimizer_convergence.png" -o \
  -name "bo_objective_landscape_best_optimizer.png" -o \
  -name "controlled_vs_uncontrolled_x.png" -o \
  -name "controlled_all_states.png" -o \
  -name "control_sweep_summary.png" -o \
  -name "pyragas_phase_trajectory_3d.png" -o \
  -name "pyragas_periodic_zoom.png" -o \
  -name "pyragas_periodic_k_sweep_summary.png" \
\) -exec cp {} "$FINAL_ROOT/07_report_figures/" \; || true

echo "========================================================================"
echo "FINAL PIPELINE COMPLETED"
echo "========================================================================"
echo "Final outputs:"
echo "$FINAL_ROOT"
echo ""
echo "Important files:"
echo "$FINAL_ROOT/00_manifest/final_manifest.json"
echo "$FINAL_ROOT/06_comparison_tables/final_controller_comparison.csv"
echo "$FINAL_ROOT/06_comparison_tables/final_controller_comparison.md"
echo "$FINAL_ROOT/00_manifest/ALL_OUTPUT_FILES.txt"

#!/bin/bash
#SBATCH --job-name=chapter1_final
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/final_thesis_pipeline_%j.out
#SBATCH --error=logs/final_thesis_pipeline_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
[[ -f "$PROJECT_DIR/final_pipeline.py" ]] || {
  echo "Submit from the Thesis repository root."
  exit 2
}
cd "$PROJECT_DIR"

if ! git diff --quiet; then
  echo "Refusing definitive run: tracked working tree changes exist."
  exit 2
fi
if ! git diff --cached --quiet; then
  echo "Refusing definitive run: staged changes exist."
  exit 2
fi
RELEVANT_PATHS=(
  '*.py'
  '*.slurm'
  'README.md'
  'CHAPTER1_METHOD_FIXES.md'
  'requirements.txt'
  'requirements*.txt'
)
RELEVANT_UNTRACKED="$(
  git ls-files --others --exclude-standard -- "${RELEVANT_PATHS[@]}"
)"
if [[ -n "$RELEVANT_UNTRACKED" ]]; then
  echo "Refusing definitive run: relevant untracked source files exist:"
  echo "$RELEVANT_UNTRACKED"
  exit 2
fi

mkdir -p logs
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/thesis_final"
export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"

JOB_ID="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
FINAL_ROOT="$HOME/Thesis_evidence_20260624/FINAL_THESIS_RUN_${JOB_ID}"
if [[ -e "$FINAL_ROOT" ]]; then
  echo "Refusing to overwrite existing evidence: $FINAL_ROOT"
  exit 2
fi

archive_job_logs() {
  local exit_status=$?
  set +e
  if [[ -d "$FINAL_ROOT/08_logs" && -n "${SLURM_JOB_ID:-}" ]]; then
    local suffix
    for suffix in out err; do
      local source_path="$PROJECT_DIR/logs/final_thesis_pipeline_${SLURM_JOB_ID}.${suffix}"
      if [[ -f "$source_path" ]]; then
        cp "$source_path" "$FINAL_ROOT/08_logs/"
      fi
    done
  fi
  exit "$exit_status"
}
trap archive_job_logs EXIT

echo "========================================================================"
echo "CHAPTER 1 DEFINITIVE ONE-PASS PIPELINE"
echo "========================================================================"
echo "Project dir : $PROJECT_DIR"
echo "Final root  : $FINAL_ROOT"
echo "Job ID      : $JOB_ID"
echo "Git commit  : $(git rev-parse HEAD)"
echo "Git branch  : $(git branch --show-current)"
echo "Started UTC : $(date -u --iso-8601=seconds)"
echo "========================================================================"

FINAL_COMMAND=(
  python
  final_pipeline.py
  --repo
  "$PROJECT_DIR"
  --final-root
  "$FINAL_ROOT"
  --clean-repository-at-start
)
"${FINAL_COMMAND[@]}"

VALIDATION_REPORT="$FINAL_ROOT/00_manifest/final_package_validation.json"
python - "$VALIDATION_REPORT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("valid") is not True:
    raise SystemExit("Final package validation did not pass.")
print("Final package validation: PASS")
PY

LINK_NAME="$PROJECT_DIR/FINAL_THESIS_RUN_${JOB_ID}"
if [[ ! -e "$LINK_NAME" ]]; then
  ln -s "$FINAL_ROOT" "$LINK_NAME"
fi

echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "Definitive package: $FINAL_ROOT"
echo "Validation report : $VALIDATION_REPORT"
echo "Completed UTC     : $(date -u --iso-8601=seconds)"
echo "========================================================================"

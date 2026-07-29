#!/bin/bash
#SBATCH --job-name=chapter1_seeds
#SBATCH --partition=work
#SBATCH --array=0-4%5
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/multiseed_%A_%a.out
#SBATCH --error=logs/multiseed_%A_%a.err

# The NumPy/SciPy ESN workflow is CPU-bound; no GPU resource is requested.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR"

[[ -f "multiseed_evaluation.py" ]] || {
  echo "multiseed_evaluation.py is missing from $PROJECT_DIR"
  exit 2
}
[[ -d "FINAL_THESIS_RUN" ]] || {
  echo "FINAL_THESIS_RUN is missing from $PROJECT_DIR"
  exit 2
}

mkdir -p logs

module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/thesis_final"

export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"

SEEDS=(42 123 456 789 2026)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

REFERENCE_RUN="${REFERENCE_RUN:-$PROJECT_DIR/FINAL_THESIS_RUN}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/Thesis_evidence_20260624/MULTISEED_EVAL_${SLURM_ARRAY_JOB_ID}}"

echo "========================================================================"
echo "CHAPTER 1 MULTISEED EVALUATION"
echo "========================================================================"
echo "Array job ID : ${SLURM_ARRAY_JOB_ID}"
echo "Array task   : ${SLURM_ARRAY_TASK_ID}"
echo "Seed         : ${SEED}"
echo "Project      : ${PROJECT_DIR}"
echo "Reference    : ${REFERENCE_RUN}"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Started UTC  : $(date -u --iso-8601=seconds)"
echo "========================================================================"

python multiseed_evaluation.py \
  --seed "$SEED" \
  --reference-run "$REFERENCE_RUN" \
  --output-root "$OUTPUT_ROOT"

echo "========================================================================"
echo "SEED ${SEED} COMPLETE"
echo "Completed UTC: $(date -u --iso-8601=seconds)"
echo "========================================================================"

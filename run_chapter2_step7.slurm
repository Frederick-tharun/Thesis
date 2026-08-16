#!/bin/bash
#SBATCH --job-name=chapter2_step7
#SBATCH --partition=rtx3080
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --output=logs/chapter2_step7_%j.out
#SBATCH --error=logs/chapter2_step7_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
[[ -f "$PROJECT_DIR/chapter2/run_bayesian_optimisation.py" ]] || {
  echo "Submit from the Thesis repository root."
  exit 2
}
cd "$PROJECT_DIR"
mkdir -p logs chapter2/optimisation_results

module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
CHAPTER2_CONDA_ENV="${CHAPTER2_CONDA_ENV:-$HOME/.conda/envs/thesis_final}"
conda activate "$CHAPTER2_CONDA_ENV"
export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

STEP7_MODEL="${STEP7_MODEL:-both}"
echo "Step 7 model scope: $STEP7_MODEL"

python3 chapter2/run_bayesian_optimisation.py \
  --model "$STEP7_MODEL" \
  --n-calls 40 \
  --resume

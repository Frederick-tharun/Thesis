#!/bin/bash
#SBATCH --job-name=ch2_cri_final
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/chapter2_cross_regime_improvement_final_%j.out
#SBATCH --error=logs/chapter2_cross_regime_improvement_final_%j.err

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR"
[[ "$(git branch --show-current)" == "chapter2-cross-regime-improvement" ]]
git diff --quiet
git diff --cached --quiet
[[ -f chapter2/cross_regime_improvement/results/optimisation/selection.json ]]
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
CHAPTER2_CONDA_ENV="${CHAPTER2_CONDA_ENV:-$HOME/.conda/envs/thesis_final}"
conda activate "$CHAPTER2_CONDA_ENV"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -u -m chapter2.cross_regime_improvement.experiment --final --resume

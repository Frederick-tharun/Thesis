#!/bin/bash
#SBATCH --job-name=ch2_cross_regime
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/chapter2_cross_regime_%j.out
#SBATCH --error=logs/chapter2_cross_regime_%j.err

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR"
[[ -f chapter2/cross_regime_results/cross_regime_manifest.json ]] || {
  echo "Submit from the repository root after freezing the protocol."
  exit 2
}
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
CHAPTER2_CONDA_ENV="${CHAPTER2_CONDA_ENV:-$HOME/.conda/envs/thesis_final}"
conda activate "$CHAPTER2_CONDA_ENV"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_DIR"
# Small 100-unit reservoirs run efficiently with a single BLAS thread.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
printf 'Cross-regime job %s on %s\n' "$SLURM_JOB_ID" "${SLURMD_NODENAME:-unknown}"
python -u -m chapter2.run_cross_regime --full
python -u -m chapter2.audit_cross_regime

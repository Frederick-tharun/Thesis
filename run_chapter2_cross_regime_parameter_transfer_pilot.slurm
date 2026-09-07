#!/bin/bash
#SBATCH --job-name=ch2_pt_pilot
#SBATCH --partition=work
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/chapter2_cross_regime_parameter_transfer_pilot_%j.out
#SBATCH --error=logs/chapter2_cross_regime_parameter_transfer_pilot_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Thesis repository root}"
[[ "$(git branch --show-current)" == "chapter2-cross-regime-parameter-transfer" ]]
git diff --quiet
git diff --cached --quiet
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/thesis_final"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -u -m chapter2.cross_regime_parameter_transfer.experiment --pilot

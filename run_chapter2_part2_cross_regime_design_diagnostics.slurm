#!/bin/bash
#SBATCH --clusters=tinyfat
#SBATCH --job-name=ch2p2_design_diag
#SBATCH --partition=work
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --output=chapter2/part2_cross_regime/results/design/slurm_%j.out
#SBATCH --error=chapter2/part2_cross_regime/results/design/slurm_%j.err

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_DIR"
mkdir -p chapter2/part2_cross_regime/results/design

module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/thesis_final"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HOME/Thesis"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 -u -m chapter2.part2_cross_regime.design_diagnostics

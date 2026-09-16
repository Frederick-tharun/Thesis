#!/bin/bash
#SBATCH --clusters=tinyfat
#SBATCH --job-name=ch2_part2_regime_map
#SBATCH --partition=work
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=chapter2/part2_regime_map/results/slurm_%j.out
#SBATCH --error=chapter2/part2_regime_map/results/slurm_%j.err

# CPU-only Hindmarsh-Rose bifurcation/regime-map sweep. No GPU is requested:
# this step is pure ODE integration, never an ESN. The default login node's
# "tinygpu" Slurm cluster forces --gres=gpu on every job, so this submits to
# the separate "tinyfat" cluster (sacctmgr show clusters), which has a
# GPU-free CPU "work" partition, matching the CPU-batch-job intent.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
[[ -f "$PROJECT_DIR/chapter2/part2_regime_map/generate_regime_map.py" ]] || {
  echo "Submit from the Thesis repository root."
  exit 2
}
cd "$PROJECT_DIR"
mkdir -p chapter2/part2_regime_map/results

module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
CHAPTER2_CONDA_ENV="${CHAPTER2_CONDA_ENV:-$HOME/.conda/envs/thesis_final}"
conda activate "$CHAPTER2_CONDA_ENV"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Stage 1/2: dense sweep + sparse Lyapunov subset"
python3 -u -m chapter2.part2_regime_map.generate_regime_map --run

echo "Stage 2/2: classification, candidate regions, bifurcation figure"
python3 -u -m chapter2.part2_regime_map.analyse_regimes

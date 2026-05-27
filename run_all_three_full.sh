#!/bin/bash
#SBATCH --job-name=hr_all_full
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e

cd /home/hpc/rlvl/rlvl177v/Thesis

module load python/3.12-conda

unset CONDA_ENVS_DIRS
unset CONDA_ENVS_PATH
export CONDA_PKGS_DIRS=/home/hpc/rlvl/rlvl177v/.conda/pkgs

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/hpc/rlvl/rlvl177v/Thesis/.conda_env

export PYTHONNOUSERSITE=1

MODES=("periodic_spiking" "periodic_bursting" "chaotic_bursting")
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Running FULL pipeline for HR mode: $MODE"
echo "This includes prediction + optimization + final model + control"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Python: $(which python)"
python --version
echo "=========================================="

python main.py --dataset hr --hr-mode "$MODE" --control --auto-control-k --control-target-mode rest_state

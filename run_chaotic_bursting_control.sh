#!/bin/bash
#SBATCH --job-name=hr_chaotic_bursting_control
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

cd /home/hpc/rlvl/rlvl177v/Thesis

module load python/3.12-conda

unset CONDA_ENVS_DIRS
unset CONDA_ENVS_PATH
export CONDA_PKGS_DIRS=/home/hpc/rlvl/rlvl177v/.conda/pkgs

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/hpc/rlvl/rlvl177v/Thesis/.conda_env

export PYTHONNOUSERSITE=1

echo "Python path: $(which python)"
python --version

python main.py --dataset hr --hr-mode chaotic_bursting --control --auto-control-k --control-target-mode rest_state

#!/bin/bash
#SBATCH --job-name=hr_periodic_spiking
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/hr_periodic_spiking_%j.out
#SBATCH --error=logs/hr_periodic_spiking_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8

cd ~/Thesis

module load python/3.12-conda
export CONDA_PKGS_DIRS=~/conda_pkgs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/Thesis/.conda_env

export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started on: $(hostname)"
echo "Start time: $(date)"
echo "Python path: $(which python)"
python --version

python main.py --dataset hr --hr-mode periodic_spiking --control --auto-control-k

echo "End time: $(date)"
echo "Job finished."
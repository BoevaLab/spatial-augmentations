#!/bin/bash

#SBATCH --job-name=phenotype_benchmark
#SBATCH --output=logs/slurm/benchmark/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/utils/benchmark_phenotype.py

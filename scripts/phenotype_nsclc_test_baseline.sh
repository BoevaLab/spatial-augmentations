#!/bin/bash

#SBATCH --job-name=phenotype_nsclc_test_baseline
#SBATCH --output=logs/slurm/phenotype_nsclc_test_baseline/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/eval_phenotype.py experiment=phenotype_nsclc_test_baseline

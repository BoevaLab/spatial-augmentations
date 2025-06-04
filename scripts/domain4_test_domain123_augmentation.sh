#!/bin/bash

#SBATCH --job-name=domain4_test_domain123_augmentation
#SBATCH --output=logs/slurm/domain4_test_domain123_augmentation/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/eval_domain.py experiment=domain4_test_domain123_augmentation

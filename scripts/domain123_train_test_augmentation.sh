#!/bin/bash

#SBATCH --job-name=domain123_train_test_augmentation
#SBATCH --output=logs/slurm/domain123_train_test_augmentation/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train_domain.py experiment=domain123_train_test_augmentation

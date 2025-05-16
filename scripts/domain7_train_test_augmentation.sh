#!/bin/bash

#SBATCH --job-name=domain7_train_test_augmentation
#SBATCH --output=logs/slurm/domain7_train_test_augmentation/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python src/train_domain.py experiment=domain7_train_test_augmentation

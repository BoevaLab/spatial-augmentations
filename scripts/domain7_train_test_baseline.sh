#!/bin/bash

#SBATCH --job-name=domain7_train_test_baseline
#SBATCH --output=logs/slurm/domain7_train_test_baseline/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train.py experiment=domain7_train_test_baseline

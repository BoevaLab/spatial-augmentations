#!/bin/bash

#SBATCH --job-name=domain123_train_test_baseline
#SBATCH --output=domain123_train_test_baseline.txt
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train.py experiment=domain123_train_test_baseline

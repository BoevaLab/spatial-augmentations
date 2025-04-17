#!/bin/bash

#SBATCH --job-name=domain4_test_domain123_baseline
#SBATCH --output=logs/slurm/domain4_test_domain123_baseline/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/eval.py experiment=domain4_test_domain123_baseline

#!/bin/bash

#SBATCH --job-name=domain123_hparam_baseline
#SBATCH --output=logs/slurm/domain123_hparam_baseline/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train_domain.py -m hparams_search=domain123_baseline experiment=domain123_hparam_baseline

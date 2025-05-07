#!/bin/bash

#SBATCH --job-name=domain123_hparam_augmentation
#SBATCH --output=logs/slurm/domain123_hparam_augmentation/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train_domain.py -m hparams_search=domain123_augmentation experiment=domain123_hparam_augmentation

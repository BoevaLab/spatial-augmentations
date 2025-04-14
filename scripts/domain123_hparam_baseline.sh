#!/bin/bash

#SBATCH --job-name=domain123_hparam_baseline
#SBATCH --output=domain123_hparam_baseline.txt
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train.py -m hparams_search=domain_baseline experiment=domain123_hparam_baseline

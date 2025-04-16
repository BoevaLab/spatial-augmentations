#!/bin/bash

#SBATCH --job-name=domain7_hparam_baseline
#SBATCH --output=logs/slurm/domain7_hparam_baseline/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train.py -m hparams_search=domain7_baseline experiment=domain7_hparam_baseline

#!/bin/bash

#SBATCH --job-name=domain7_hparam_baseline
#SBATCH --output=logs/slurm/domain7_hparam_baseline/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python src/train.py -m hparams_search=domain7_baseline experiment=domain7_hparam_baseline

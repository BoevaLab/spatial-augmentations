#!/bin/bash

#SBATCH --job-name=phenotype_nsclc_hparam_baseline
#SBATCH --output=logs/slurm/phenotype_nsclc_hparam_baseline/%j.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train_phenotype.py -m hparams_search=phenotype_nsclc_baseline experiment=phenotype_nsclc_hparam_baseline

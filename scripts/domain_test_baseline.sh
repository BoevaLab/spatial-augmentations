#!/bin/bash

#SBATCH --job-name=domain_test_baseline
#SBATCH --output=output.txt
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00

# shellcheck disable=SC1091
source "$HOME/.bashrc"
conda activate augmentation

srun python src/train.py tags="[baseline, test]" \
    ckpt_path=??? \
    data.batch_size=1 \
    data.num_workers=1 \
    trainer=gpu \

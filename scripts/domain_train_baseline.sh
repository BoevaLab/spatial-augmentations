#!/bin/bash

#SBATCH --job-name=domain_train_baseline
#SBATCH --output=output.txt
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

source ~/.bashrc
conda activate augmentation

srun python src/train.py tags="[baseline, train]" \
    data.batch_size=4 \
    data.num_workers=4 \
    trainer=gpu \
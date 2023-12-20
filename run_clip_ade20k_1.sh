#!/bin/bash
#SBATCH --job-name=run_clip_ade20k_1
#SBATCH --output=run_clip_ade20k_1.out
#SBATCH --error=run_clip_ade20k_1.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1

python run_clip_ade20k.py --loss_scale 1
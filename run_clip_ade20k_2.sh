#!/bin/bash
#SBATCH --job-name=run_clip_ade20k_2
#SBATCH --output=run_clip_ade20k_2.out
#SBATCH --error=run_clip_ade20k_2.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1

python run_clip_ade20k.py --loss_scale 2
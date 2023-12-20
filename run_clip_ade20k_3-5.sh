#!/bin/bash
#SBATCH --job-name=run_clip_ade20k_3-5
#SBATCH --output=run_clip_ade20k_3-5.out
#SBATCH --error=run_clip_ade20k_3-5.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1

python run_clip_ade20k.py --loss_scale 3.5
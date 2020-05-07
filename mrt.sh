#!/bin/bash
#SBATCH -c 6 # Number of cores
#SBATCH -N 1 # 1 node requested
#SBATCH -n 1 # 1 task requested
#SBATCH --mem=32000 # Memory - Use 32G
#SBATCH --time=0 # No time limit
#SBATCH -o mrt-30.slurm  # send stdout to outfile
#SBATCH --gres=gpu:1 # Use 1 GPUs
#SBATCH -p gpu

python myTrain.py \
  -dec=TRADE -bsz=16 -dr=0.2 -lr=0.0003 -le=1 \
  --mrt --addName mrt-30 --use_ont \
  -path=save/TRADE-multiwozdst/HDD400BSZ32DR0.2ACC-0.4970/
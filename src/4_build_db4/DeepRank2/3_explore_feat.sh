#!/bin/bash
#SBATCH --job-name 3_explore_feat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/logs/deeprank2/3_explore_feat_job-%J.out

source activate deeprank2_gpu

python -u 3_explore_feat.py

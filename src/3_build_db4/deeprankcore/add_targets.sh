#!/bin/bash
#SBATCH --job-name add_targets
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/logs/deeprankcore/add_targets_job-%J.out

source activate deeprank_gpu

python -u add_targets.py

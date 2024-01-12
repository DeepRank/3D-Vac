#!/bin/bash
#SBATCH --job-name add_targets
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 72
#SBATCH --time 04:00:00
#SBATCH -o /projects/0/einf2380/data/logs/deeprank2/add_targets_job-%J.out

source activate deeprank2_gpu
python -u add_targets.py

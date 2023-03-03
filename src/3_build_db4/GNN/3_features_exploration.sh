#!/bin/bash
#SBATCH --job-name features_exploration
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 06:00:00
#SBATCH --partition fat
#SBATCH -o /projects/0/einf2380/data/training_logs/I/features_exploration_GNN_residue_job-%J.out

source activate deeprank_gpu

python -u 3_features_exploration.py

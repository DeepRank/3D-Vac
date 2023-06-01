#!/bin/bash
#SBATCH --job-name deeprankcore_training_GNN
#SBATCH --partition gpu
#SBATCH --gpus 1
# #SBATCH --partition fat
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --time 30:00:00
#SBATCH -o /projects/0/einf2380/data/logs/deeprankcore/deeprankcore_training_GNN-%J.out

source activate deeprank_gpu
python -u training.py

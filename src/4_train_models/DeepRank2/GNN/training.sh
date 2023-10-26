#!/bin/bash
#SBATCH --job-name deeprank2_training_GNN
#SBATCH --partition gpu
#SBATCH --gpus 1
# #SBATCH --partition fat
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --time 20:00:00
#SBATCH -o /projects/0/einf2380/data/logs/deeprank2/deeprank2_training_GNN-%J.out

source activate deeprank2_gpu
python -u training.py

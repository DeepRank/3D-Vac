#!/bin/bash
#SBATCH --job-name training_GNN
#SBATCH --partition gpu
#SBATCH --gpus 1
# #SBATCH --partition fat
#SBATCH --nodes 1
#SBATCH --ntasks 1
# #SBATCH --cpus-per-task 36
#SBATCH --time 1:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/training_GNN_residue_job-%J.out

source activate deeprank_gpu
python -u training.py

#!/bin/bash
#SBATCH --job-name 0_generate_hdf5_GNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=1-00:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/training_logs/I/generate_hdf5_GNN_residue_job-%J.out

source activate deeprank_gpu

python -u 1_generate_features.py

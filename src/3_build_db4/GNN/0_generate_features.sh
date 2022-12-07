#!/bin/bash
#SBATCH --job-name 0_generate_hdf5_GNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --partition=fat

source activate deeprank

python -u 0_generate_features.py

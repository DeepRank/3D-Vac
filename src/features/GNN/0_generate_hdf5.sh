#!/bin/bash
#SBATCH --job-name 0_generate_hdf5_GNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=thin
#SBATCH --time=01:00:00
#SBATCH -o /projects/0/einf2380/data/preproc_logs/0_generate_hdf5_GNN_job-%J.out

source activate deeprank

python -u 0_generate_hdf5.py
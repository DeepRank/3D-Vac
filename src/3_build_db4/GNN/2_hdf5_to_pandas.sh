#!/bin/bash
#SBATCH --job-name hdf5_to_pandas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/training_logs/I/hdf5_to_pandas_GNN_residue_job-%J.out

source activate deeprank

python -u 2_hdf5_to_pandas.py

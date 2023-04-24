#!/bin/bash
#SBATCH --job-name hdf5_to_pandas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/training_logs/I/feat_pandas_hist_GNN_residue_job-%J.out

source activate deeprank_gpu

python -u 2_feat_pandas_hist.py

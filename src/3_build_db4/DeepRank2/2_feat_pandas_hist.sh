#!/bin/bash
#SBATCH --job-name 2_feat_pandas_hist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/logs/deeprank2/2_feat_pandas_hist_job-%J.out

source activate deeprank2_gpu

python -u 2_feat_pandas_hist.py

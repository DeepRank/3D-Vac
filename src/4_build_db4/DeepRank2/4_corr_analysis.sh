#!/bin/bash
#SBATCH --job-name 4_corr_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=10:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/logs/deeprank2/4_corr_analysis_job-%J.out

source activate deeprank2_gpu

python -u 4_corr_analysis.py

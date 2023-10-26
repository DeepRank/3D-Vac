#!/bin/bash
#SBATCH --job-name 1_generate_features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=20:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/logs/deeprank2/1_generate_features_job-%J.out

source activate deeprank2_gpu

python -u 1_generate_features.py

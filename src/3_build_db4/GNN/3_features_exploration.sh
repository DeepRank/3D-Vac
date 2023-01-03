#!/bin/bash
#SBATCH --job-name features_exploration
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --time 06:00:00
#SBATCH --partition fat

source activate deeprank

python -u 3_features_exploration.py

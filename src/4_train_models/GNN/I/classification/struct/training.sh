#!/bin/bash
#SBATCH --job-name training_GNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --partition=fat
#SBATCH -o /projects/0/einf2380/data/preproc_logs/training_GNN_residue_job-%J.out
source activate deeprank
python -u training.py
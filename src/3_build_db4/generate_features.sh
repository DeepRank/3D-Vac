#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --partition=fat
#SBATCH --time=01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/generate_features.out

# Load modules:
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0

# Activate conda environment:
source /home/lepikhovd/.bashrc
source activate deeprank

# Start the script:
srun python -u generate_features.py
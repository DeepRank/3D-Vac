#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH --cpus-per-task 128
#SBATCH --time 03:00:00

# Load modules for MPI:
module load 2021
module load foss/2021a

# Load conda environment
source activate deeprank

# Usage: srun python -u clean_outputs.py -p <path_to_models/*/*>
python clean_outputs.py "$@"
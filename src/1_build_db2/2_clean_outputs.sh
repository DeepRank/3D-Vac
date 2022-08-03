#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 128
#SBATCH --time 01:00:00
#SBATCH --export=NONE
#SBATCH -o /projects/0/einf2380/data/modelling_logs/clean_models_job.out

# Load modules for MPI:
module load 2021
module load foss/2021a

# Load conda environment
source activate deeprank

# Usage: srun python -u clean_outputs.py -p <path_to_models/*/*>
srun python -u clean_outputs.py "$@"
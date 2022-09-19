#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH --cpus-per-task 128
#SBATCH --time 01:00:00
#SBATCH --export=NONE # dont take environment variables from the user
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/clean_models_job-%J.out
#SBATCH -e /projects/0/einf2380/data/modelling_logs/I/clean_models_job-%J.err

# Load modules for MPI:
module load 2021
module load foss/2021a

# Load conda environment
source activate deeprank

# Usage: srun python -u clean_outputs.py -p <path_to_models/*/*>
python -u clean_outputs.py "$@"
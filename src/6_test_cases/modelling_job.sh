#!/bin/bash
#SBATCH -p thin
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name modelling
#SBATCH --no-kill
#SBATCH --time=00:20:00

## usage: srun python modelling_job.py <running time>

source activate deeprank

srun  python -u model_test_cases.py
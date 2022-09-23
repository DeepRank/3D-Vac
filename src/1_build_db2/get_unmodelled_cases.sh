#!/bin/bash
#SBATCH -p thin
#SBATCH --job-name check_unmodelled
#SBATCH --cpus-per-task 64
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/unmodelled_logs-%J.out
#SBATCH --time=0-00:45:00

## load modules
source activate deeprank
## usage: python get_unmodelled_cases.py
srun python get_unmodelled_cases.py "$@"
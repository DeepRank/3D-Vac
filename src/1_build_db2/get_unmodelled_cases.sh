#!/bin/bash
#SBATCH -p thin
#SBATCH --job-name check_unmodelled
#SBATCH --cpus-per-task 64
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db2/unmodelled_logs-%J.out
#SBATCH --time=01:00:00

## load modules
source activate deeprank
## usage: python get_unmodelled_cases.py
python get_unmodelled_cases.py "$@"
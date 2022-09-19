#!/bin/bash
#SBATCH -p thin
#SBATCH --job-name check_unmodelled
#SBATCH --cpus-per-task 128
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/unmodelled_logs-%J.out
#SBATCH -e /projects/0/einf2380/data/modelling_logs/I/unmodelled_logs-%J.err
#SBATCH --time=0-00:45:00

## load modules
source activate deeprank
## usage: python get_unmodelled_cases.py
python -u get_unmodelled_cases.py "$@"
#!/bin/bash
#SBATCH -p thin
#SBATCH --job-name check_unmodelled
#SBATCH -o /projects/0/einf2380/data/modelling_logs/unmodelled_logs-%J.out
#SBATCH -e /projects/0/einf2380/data/modelling_logs/unmodelled_logs-%J.err

## load modules
source activate deeprank
## usage: python get_unmodelled_cases.py
python get_unmodelled_cases.py "$@"
#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes 1
#SBATCH --job-name check_unmodelled
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/unmodelled_logs.out

## load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
## usage: python get_unmodelled_cases.py
python get_unmodelled_cases.py
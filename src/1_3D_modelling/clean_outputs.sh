#!/bin/bash
#SBATCH -p thin
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --time 01:00:00
#SBATCH --export=NONE
#SBATCH -o /projects/0/einf2380/data/modelling_logs/clean_outputs.out

source activate python2
mpiexec -n 128 python ./clean_outputs.py
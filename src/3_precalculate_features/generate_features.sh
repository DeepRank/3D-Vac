#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --partition=fat
#SBATCH --time=01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/generate_features.out

source /home/lepikhovd/.bashrc
source activate deeprank

mpiexec -n 32 python -u generate_features.py
#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/map_pssm2pdb.out

mpiexec -n 128 python -u ./map_pssm2pdb.py
#!/bin/bash
#SBATCH --job-name get_propedia_alleles
#SBATCH --partition thin
#SBATCH -o /home/dmarz/3D-Vac/src/EGNN/%J.out
#SBATCH -e /home/dmarz/3D-Vac/src/EGNN/%J.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00


source activate deeprank

python get_propedia_alleles.py

    
    

#!/bin/bash
#SBATCH --job-name pdb_to_dhf5_gnns
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=thin
#SBATCH --time=01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/pdb_to_dhf5_gnns_job-%J.out

source activate deeprank

python -u pdb_to_hdf5_gnns.py
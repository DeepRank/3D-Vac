#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --partition=thin
#SBATCH --time=01:00:00

python cluster_peptides.py -e -u -l 15 -n 64 -f ../../data/external/processed/IDs_BA_DRB0101_MHCII_15mers.csv

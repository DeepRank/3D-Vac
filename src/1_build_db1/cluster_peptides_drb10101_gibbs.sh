#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --partition=thin
#SBATCH --time=01:00:00

python cluster_peptides.py -g -e -l 15 -n 64 -f /projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII.csv

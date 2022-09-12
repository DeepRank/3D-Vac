#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/map_pssm2pdb_I.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u map_pssm2pdb.py --mhc-class I --csv-file ../../data/external/processed/BA_pMHCII.csv
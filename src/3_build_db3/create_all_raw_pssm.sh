#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/create_raw_pssm_I.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u create_raw_pssm.py --mhc-class I \ 
    --input-csv /projects/0/einf2380/data/external/processed/I/IDs_BA_MHCI.csv
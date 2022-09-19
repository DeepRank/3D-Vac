#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/map_pssm2pdb_II.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun --job-name map_pssm \
    python -u map_pssm2pdb.py  --mhc-class II \ 
    --csv-file /projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv
srun --dependency=afterany:map_pssm \
    merge_pdbs_and_pssms_chains.py
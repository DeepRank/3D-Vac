#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/map_pssm2pdb_I.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u map_pssm2pdb.py --mhc-class I \
    --csv-file /projects/0/einf2380/data/external/processed/I/IDs_BA_MHCI.csv \
    --alphachain_pssm /projects/0/einf2380/data/pMHCII/pssm_raw/hla_02_01/pssm_raw/hla_02_01.pssm #WARNING: please generate this pssm, do not use the Human_MHC_data.pssm present already
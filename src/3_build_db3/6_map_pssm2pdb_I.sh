#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time 02:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db3/map_pssm2pdb_I-%J.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u map_pssm2pdb.py --mhc-class I \
    --csv-file /projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative.csv \
    --alphachain-pssm /projects/0/einf2380/data/pMHCI/pssm_raw/all_mhc #WARNING: please generate this pssm, do not use the Human_MHC_data.pssm present already
#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/map_pssm2pdb_II.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

# raw_pssm_N = "/projects/0/einf2380/data/pMHCII/pssm_raw/hla_drb1_0101/pssm_raw/hla_drb1_0101.pssm"
# raw_pssm_M = "/projects/0/einf2380/data/pMHCII/pssm_raw/hla_drb1_0101/pssm_raw/hla_dra_0101.pssm"

srun --job-name map_pssm \
    python -u map_pssm2pdb.py --mhc-class II \
    --csv-file /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII_noduplicates.csv \
    --alphachain-pssm /projects/0/einf2380/data/pMHCII/pssm_raw/all_mhc \
    --betachain-pssm /projects/0/einf2380/data/pMHCII/pssm_raw/all_mhc

#srun --dependency=afterany:map_pssm \
#    python merge_pdbs_and_pssms_chains.py \
#    --input-folders /projects/0/einf2380/data/pMHCII/db2_selected_models/BA/\*/\*
#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH --cpus-per-task 128
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/II/copy_3Dmodels_from_db2_II-%J.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u ./copy_3Dmodels_from_db2.py --mhc-class II \
    --csv-file /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII_noduplicates.csv \
    --models-path /projects/0/einf2380/data/pMHCII/3d_models/BA/\*/\* \
    --best-models-path /projects/0/einf2380/data/pMHCII/db2_selected_models
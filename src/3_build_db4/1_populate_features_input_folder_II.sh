#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#removed -c 1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/II/populate_input_folder.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u populate_features_input_folder.py \
    --input-folder /projects/0/einf2380/data/pMHCII/db2_selected_models/BA/\*/\* \
    --output-folder /projects/0/einf2380/data/pMHCII/features_input_folder/hla_drb1_0101_15mers
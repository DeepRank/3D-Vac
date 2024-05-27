#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/populate_input_folder.out
#SBATCH -e /projects/0/einf2380/data/modelling_logs/I/populate_input_folder.err

# Load modules:
module load 2022
module load foss/2022a

# Activate conda env:
source activate deeprank

srun python -u populate_features_input_folder.py \
    --input-folder /projects/0/einf2380/data/pMHCI/db2_selected_models/BA/\*/\* \
    --output-folder /projects/0/einf2380/data/pMHCI/features_input_folder/exp_nmers_all_HLA_quantitative
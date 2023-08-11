#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=thin
#SBATCH --time=04:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/add_features.out
#SBATCH -e /projects/0/einf2380/data/modelling_logs/add_features.err

# Activate conda environment:
source activate deeprank
#$TARGET_INPUT_CSV='/projects/0/einf2380/3D-Vac/data/external/processed/BA_pMHCI.csv'

# Start the script:
srun python -u add_feature.py \
    --data-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative/ \
    --features-name sequence_feature
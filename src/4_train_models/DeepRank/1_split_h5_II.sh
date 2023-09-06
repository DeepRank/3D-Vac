#!/bin/bash
#removed --job-name split_h5
#removed -o /projects/0/einf2380/data/training_logs/split_h5_job-%J.out

# Usage: bash 1_split_h5_II.sh

source activate deeprank
python -u split_h5.py --features-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15mers/original/ \
    --output-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15mers/ \

python -u split_h5.py --features-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15mers/original/ \
    --cluster --csv-file /projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv \
    --output-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15mers/ \
    
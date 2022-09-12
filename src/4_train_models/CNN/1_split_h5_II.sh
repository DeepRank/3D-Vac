#!/bin/bash
#SBATCH --job-name split_h5
#SBATCH -o /projects/0/einf2380/data/training_logs/split_h5_job-%J.out

source activate deeprank
python -u split_h5.py --features-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15_mers/ \
    --output-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15_mers/splits \

python -u split_h5.py --features-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15_mers/ \
    --cluster --csv-file ../../../data/external/processed/BA_pMHCII.csv \
    --output-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15_mers/splits \
    
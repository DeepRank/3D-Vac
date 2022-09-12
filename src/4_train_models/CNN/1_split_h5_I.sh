#!/bin/bash
#SBATCH --job-name split_h5
#SBATCH -o /projects/0/einf2380/data/training_logs/split_h5_job-%J.out

source activate deeprank
python -u split_h5.py --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/ \
    --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/splits \

python -u split_h5.py --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/ \
    --cluster --csv-file ../../../data/external/processed/BA_pMHCI.csv \
    --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/splits \
    
    

#!/bin/bash
#SBATCH --job-name split_h5
#SBATCH --partition thin
#SBATCH -o /projects/0/einf2380/data/training_logs/split_h5_job-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/split_h5_job-%J.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00

# Usage: sbatch 1_split_h5_I.sh

source activate deeprank

# #Shuffled
# python ./split_h5.py \
#     --csv-file /projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative_only_eq.csv \
#     --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative/ \
#     --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled \
#     --single-split \
#     --parallel 64

EXPERIMENT=Shuffled
for FOLD_NUMBER in $(seq 1 5); do
    python split_h5.py  --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative/ \
        --train-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/$FOLD_NUMBER/train.csv \
        --val-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/$FOLD_NUMBER/validation.csv \
        --test-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/test.csv \
        --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/CrossValidation/$EXPERIMENT/$FOLD_NUMBER \
        --single-split \
        --parallel 64
done

#Allele Clustered
EXPERIMENT=AlleleClustered
for FOLD_NUMBER in $(seq 1 5); do
    python split_h5.py  --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative/ \
        --train-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/$FOLD_NUMBER/train.csv \
        --val-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/$FOLD_NUMBER/validation.csv \
        --test-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/test.csv \
        --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/CrossValidation/$EXPERIMENT/$FOLD_NUMBER \
        --single-split \
        --parallel 64
done

    
    

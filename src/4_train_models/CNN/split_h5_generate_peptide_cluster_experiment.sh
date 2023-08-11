#!/bin/bash

#SBATCH --partition thin
#SBATCH --time 1:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --output /projects/0/einf2380/data/modelling_logs/I/db4/split_h5-%J.out 

source activate deeprank

python ./split_h5.py \
    --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative/ \
    --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/cluster_4_as_test_exp \
    --parallel \
    --trainval-csv "/projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_peptide_clustered_train_validation.csv" \
    --test-csv "/projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_peptide_clustered_test.csv"

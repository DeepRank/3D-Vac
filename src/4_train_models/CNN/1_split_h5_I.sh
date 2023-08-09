#!/bin/bash
#SBATCH --job-name split_h5
#SBATCH --partition thin
#SBATCH -o /projects/0/einf2380/data/training_logs/split_h5_job-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/split_h5_job-%J.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00

# Usage: bash 1_split_h5_I.sh

source activate deeprank

# python ./split_h5.py \
# --csv-file /projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative_all_hla_gibbs_clusters.csv \
# --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative_backup/ \
# --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/all_HLA_quant_shuffled_peptides_10fold \
# --n-fold 10 \
# --parallel 128


python ./split_h5.py \
--csv-file /projects/0/einf2380/data/external/processed/I/Marieke_10C_BA_pMHCI_human_quantitative.csv \
--features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative/ \
--output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/split_HLA_quant_Marieke_clust \
--cluster \
--cluster-column cluster \
--test-clusters 3 \
--parallel 64
#--train-clusters 0 2 3 4 5 6 7 8 9 \

    
    

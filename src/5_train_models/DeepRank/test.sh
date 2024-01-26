#!/bin/bash
#SBATCH --job-name split_h5
#SBATCH -o /projects/0/einf2380/data/training_logs/test-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/test-%J.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00

# Usage: bash 1_split_h5_I.sh

source activate 3D-Vac

# python ./split_h5.py \
# --csv-file /projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative_all_hla_gibbs_clusters.csv \
# --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative_backup/ \
# --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/all_HLA_quant_shuffled_peptides_10fold \
# --n-fold 10 \
# --parallel 128


python ./test.py

    
    

#!/bin/bash

#SBATCH --nodes 1
#SBATCH --partition thin
#SBATCH --time 10:00:00
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 2
#SBATCH -o /projects/0/einf2380/data/training_logs/I/MLP/training_MLP_clustered_peptides_exp-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/MLP/training_MLP_clustered_peptides_exp-%J.err

module load 2021
module load foss/2021a

source activate mlp

srun python -u /home/dmarz/3D-Vac/src/4_train_models/seq/I/mlp_baseline.py \
    --csv-file /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_peptide_clustered_train_validation.csv \
    --test-csv /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_peptide_clustered_test.csv \
    --encoder blosum_with_allele \
    --trained-models-path projects/0/einf2380/data/pMHCI/trained_models/MLP_rerun/shuffled \
    --neurons 500 \
    --batch 64 \
    --epochs 50 \
    --model-name exp_clustered_peptides \
    --task regression

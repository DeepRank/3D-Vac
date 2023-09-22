#!/bin/bash

#SBATCH --nodes 1
#SBATCH --partition fat
#SBATCH --time 1:00:00
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 2
#SBATCH --output /projects/0/einf2380/data/training_logs/I/training_MLP_shuffled_exp-%J.out

module load 2021
module load foss/2021a

source activate deeprank

srun python -u ../../mlp_baseline.py \
    --csv-file /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv \
    --test-csv /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_test.csv \
    --trained-models-path projects/0/einf2380/data/pMHCI/trained_models/MLP \
    --encoder blosum_with_allele \
    --neurons 500 \
    --batch 64 \
    --epochs 50 \
    --model-name exp_shuffled \
    --task regression

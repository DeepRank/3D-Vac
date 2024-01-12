#!/bin/bash
#SBATCH --partition thin
#SBATCH --nodes 1
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 12
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/MLP/mlp_classification_seq_cpu-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/MLP/mlp_classification_seq_cpu-%J.err

## load modules
source activate mlp
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0
# usage: srun python -u mlp_reg_baseline.py <arguments for the script>

srun python -u mlp_baseline.py \
    --csv-file /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv \
    --test-csv /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_test.csv \
    --trained-models-path /projects/0/einf2380/data/pMHCI/trained_models/MLP_rerun/shuffled \


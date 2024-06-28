#!/bin/bash
#SBATCH --time 06:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/MLP/train_new_mlp-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/MLP/train_new_mlp-%J.err
#SBATCH --partition gpu   
#SBATCH --gpus-per-node=1 
#SBATCH --nodes 1         
#SBATCH --ntasks 1        
#SBATCH --cpus-per-task=18


source activate mlp

EXPERIMENT=$1
FOLD_NUMBER=$2

srun python -u /home/dmarz/3D-Vac/src/5_train_models/seq/MLP.py \
    --train-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/$FOLD_NUMBER/train.csv \
    --valid-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/$FOLD_NUMBER/validation.csv \
    --test-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/test.csv \
    --trained-models-path /projects/0/einf2380/data/pMHCI/trained_models/MLP/CrossValidations \
    --encoder blosum_with_allele \
    --neurons 500 \
    --batch 64 \
    --epochs 200 \
    --model-name MLP \
    --task regression \
    --experiment ${EXPERIMENT}_${FOLD_NUMBER} \
    --clean

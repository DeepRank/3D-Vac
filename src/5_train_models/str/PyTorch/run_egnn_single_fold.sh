#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=18
#SBATCH --time 04:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/EGNN/egnn_train-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/EGNN/egnn_train-%J.err

source activate egnn

EXPERIMENT=$1
FOLD_NUMBER=$2

python train.py --train-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/$FOLD_NUMBER/train.csv \
    --val-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/$FOLD_NUMBER/validation.csv \
    --test-csv /projects/0/einf2380/data/external/processed/I/CrossValidations/$EXPERIMENT/test.csv \
    --experiment ${EXPERIMENT}_${FOLD_NUMBER} --fold $FOLD_NUMBER
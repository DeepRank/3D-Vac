#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=18
#SBATCH --time 04:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/3DSSL/ssl_train-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/3DSSL/ssl_train-%J.err

source activate egnn

EXPERIMENT=$1
FOLD_NUMBER=$2

python train_ssl.py --data-choice $EXPERIMENT --seed $FOLD_NUMBER
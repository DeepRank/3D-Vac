#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=18
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/EGNN/egnn_FullDB_train-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/EGNN/egnn_FullDB_train-%J.err

## Usage: sbatch --job-name=HBV_testcase run_egnn_hbv_training.sh

source activate egnn

python train.py --train-csv /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv \
    --val-csv /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_test.csv \
    --test-csv /home/dmarz/3D-Vac/src/6_test_cases/hbv_test_cases.csv \
    --experiment HBV_testcase --run-test False

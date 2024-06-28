#!/bin/bash
#SBATCH --job-name split_h5
#SBATCH --partition thin
#SBATCH -o /projects/0/einf2380/data/test_logs/test_erasmusmcData-%J.out
#SBATCH -e /projects/0/einf2380/data/test_logs/test_erasmusmcData-%J.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00


source activate dr2
python -u pre-trained_testing.py
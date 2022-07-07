#!/bin/bash
#SBATCH --job-name split_h5
#SBATCH -o /projects/0/einf2380/data/training_logs/split_h5_job-%J.out

source activate deeprank
python -u ./split_h5.py "$@"
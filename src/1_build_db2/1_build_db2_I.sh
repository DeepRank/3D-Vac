#!/bin/bash
#SBATCH --job-name build_db2_I
#SBATCH -o /projects/0/einf2380/data/modelling_logs/build_db2_I.out
#SBATCH -n 1
#SBATCH -c 1

source activate deeprank

python -u build_db2.py -i ../../data/external/processed/I/BA_pMHCI.csv \
    -m /projects/0/einf2380/data/pMHCI/models/BA \
    -c I \
    -n 10

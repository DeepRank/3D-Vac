#!/bin/bash
#SBATCH --job-name build_db2_I
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db2/build_db2_I-%J.out
#SBATCH -n 1
#SBATCH -c 1

source activate deeprank

python -u build_db2.py --input-csv /projects/0/einf2380/data/external/processed/I/BA_pMHCI.csv \
    --models-dir /projects/0/einf2380/data/pMHCI/models/BA_1/\*/\* \
    --mhc-class I \
    --num-nodes 10
    
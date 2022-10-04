#!/bin/bash
#SBATCH --job-name build_db2_II
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db2/build_db2_II.out
#SBATCH -n 1
#SBATCH -c 1

source activate deeprank

python -u build_db2.py -i /projects/0/einf2380/data/external/processed/II/BA_pMHCII.csv \
    --models-dir \
    --mhc-class II \
    --num-nodes 10

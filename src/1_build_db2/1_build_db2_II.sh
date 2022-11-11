#!/bin/bash
#SBATCH --job-name build_db2_II
#SBATCH -o /projects/0/einf2380/data/modelling_logs/II/db2/build_db2_II-%J.out
#SBATCH -n 1
#SBATCH -c 1

source activate deeprank

python -u build_db2.py --input-csv /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII_noduplicates.csv \
    --models-dir /projects/0/einf2380/data/pMHCII/3D_models/BA/\*/\* \
    --mhc-class II \
    --num-nodes 10

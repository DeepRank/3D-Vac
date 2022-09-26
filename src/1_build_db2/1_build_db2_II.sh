#!/bin/bash
#SBATCH --job-name build_db2_II
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/build_db2_II.out
#SBATCH -n 1
#SBATCH -c 1

source activate deeprank

python -u build_db2.py -i ../../data/external/processed/II/BA_pMHCII.csv \
    -m /projects/0/einf2380/data/pMHCII/3D_models/BA \
    -c II

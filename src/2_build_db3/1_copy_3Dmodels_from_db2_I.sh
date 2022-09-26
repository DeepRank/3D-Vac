#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH -n 128 # number of tasks
#SBATCH -c 1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/copy_3Dmodels_from_db2_I.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u ./copy_3Dmodels_from_db2.py  --mhc-class I \
     --csv-file /home/severin/3D-Vac/data/external/processed/I/BA_pMHCI.csv \
     --models-path /projects/0/einf2380/data/pMHCI/3D_models/BA_1/\*/\*
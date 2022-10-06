#!/bin/bash
#SBATCH -p thin
#SBATCH -N 1
#SBATCH --cpus-per-task 128
#SBATCH --time 01:30:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db3/copy_3Dmodels_from_db2_I-%J.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python ./copy_3Dmodels_from_db2.py  --mhc-class I \
     --csv-file /projects/0/einf2380/data/external/processed/I/BA_pMHCI_all.csv \
     --models-path /projects/0/einf2380/data/pMHCI/models/BA_1/\*/\*
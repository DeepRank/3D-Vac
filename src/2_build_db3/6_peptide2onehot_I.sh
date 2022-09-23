#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/peptide2onehot_I.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u ./peptide2onehot.py  --mhc-class I \ 
    --input-csv /projects/0/einf2380/data/external/processed/I/IDs_BA_MHCI.csv
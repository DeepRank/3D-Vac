#!/bin/bash
#SBATCH -p thin
#SBATCH --job-name peptide2onehot
#SBATCH --nodes 1
#SBATCH --cpus-per-task 128
#SBATCH --time 02:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db3/peptide2onehot_I-%J.out

# Activate conda env:
source activate deeprank

srun python ./peptide2onehot.py  --mhc-class I --input-csv /projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative.csv --models-dir /projects/0/einf2380/data/pMHCI/db2_selected_models_1/BA/\*
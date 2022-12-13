#!/bin/bash
#SBATCH -p thin
#SBATCH --job-name peptide2onehot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/peptide2onehot_II.out

# Activate conda env:
source activate deeprank

srun python -u ./peptide2onehot.py --mhc-class II \
     --input-csv /projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv
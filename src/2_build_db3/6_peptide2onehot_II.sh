#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/peptide2onehot_II.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

srun python -u ./peptide2onehot.py --mhc-class II \
     --input-csv /projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv
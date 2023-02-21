#!/bin/bash
#SBATCH -p thin
#SBATCH -n 1 -c 128
#SBATCH --job-name check_step2
#SBATCH --time 10:00:00
#SBATCH --export ALL

## load modules
#module load 2020
#module load OpenMPI/4.0.3-GCC-9.3.0
## usage: sbatch clip_C_domain_mhcII.sh
srun python -u ./check_step2_completion.py -d '/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/*/*' -i '/projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv' -o './db2_checks.tsv' -n 128
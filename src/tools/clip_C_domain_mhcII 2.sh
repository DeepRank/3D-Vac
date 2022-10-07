#!/bin/bash
#SBATCH -p thin
#SBATCH -n 1 -c 128
#SBATCH --job-name clip_C_domain
#SBATCH --time 10:00:00
#SBATCH --export ALL

## load modules
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0
## usage: sbatch clip_C_domain_mhcII.sh
srun python -u ./clip_C_domain_mhcII.py 128
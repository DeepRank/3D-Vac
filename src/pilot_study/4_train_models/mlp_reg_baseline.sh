#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes 1
#SBATCH --njobs 10
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/job-%J.out
#SBATCH --export ALL
#SBATCH -n 1 -c 128

## load modules
source activate deeprank

# usage: mpiexec -n <number of jobs> python -u mlp_reg_baseline.py <arguments for the script>
mpiexec -n 10 python -u mlp_reg_baseline.py "$@"
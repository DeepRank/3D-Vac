#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes 2
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 24
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/job-%J.out

## load modules
source activate deeprank

# usage: mpiexec -n <number of jobs> python -u mlp_reg_baseline.py <arguments for the script>
mpiexec -n 10 python -u mlp_reg_baseline.py "$@"
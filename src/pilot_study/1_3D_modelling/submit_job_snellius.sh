#!/bin/bash
#SBATCH -p thin
#SBATCH --ntasks-per-node 1
#SBATCH -o /projects/0/einf2380/data/modelling_logs/job-%J.out
#SBATCH --exclusive
#SBATCH --cpus-per-task 128

## the number of nodes for the job is provided in the dispatch_modeling_jobs_parallelized.py script
## usage: mpiexec -n [number of tasks] python ./run_wrapper.py [running time]

source activate deeprank
mpiexec -n $1 python -u ./run_wrapper_parallelized.py $2
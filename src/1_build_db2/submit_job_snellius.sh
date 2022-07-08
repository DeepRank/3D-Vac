#!/bin/bash
#SBATCH -p thin
#SBATCH --ntasks-per-node 1
#SBATCH --job-name modelling
#SBATCH -o /projects/0/einf2380/data/modelling_logs/MHCI_job-%J.out
#SBATCH --exclusive
#SBATCH --cpus-per-task 128

## the number of nodes for the job is provided in the dispatch_modeling_jobs_parallelized.py script
## usage: srun python ./run_wrapper.py <running time>

module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0

source activate deeprank
srun python -u ./run_wrapper.py $1
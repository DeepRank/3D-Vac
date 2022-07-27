#!/bin/bash
#SBATCH -p thin
#SBATCH --ntasks-per-node 1
#SBATCH --job-name modelling
#SBATCH -o /projects/0/einf2380/data/modelling_logs/MHCI_job-%J.out
#SBATCH --exclusive
#SBATCH --cpus-per-task 128

## the number of nodes for the job is provided in the dispatch_modeling_jobs_parallelized.py script
## usage: srun python modelling_job.py <running time>

module load 2021
module load foss/2021a

source activate deeprank

srun --wait=0 python -u modelling_job.py $1
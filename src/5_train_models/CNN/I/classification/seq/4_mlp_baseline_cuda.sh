#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 7
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/mlp_classification_seq_cuda-%J.out

## load modules
source activate deeprank
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0
# usage: srun python -u mlp_reg_baseline.py <arguments for the script>

srun python -u 4_mlp_baseline.py "$@"

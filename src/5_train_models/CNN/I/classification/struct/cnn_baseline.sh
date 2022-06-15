#!/bin/bash
#SBATCH --partition thin
#SBATCH --nodes 1
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 10
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/cnn_classification_struct_cpu-%J.out

# usage: srun python -u mlp_reg_baseline.py <arguments for the script>
source activate deeprank
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0
srun python -u cnn_baseline.py "$@"

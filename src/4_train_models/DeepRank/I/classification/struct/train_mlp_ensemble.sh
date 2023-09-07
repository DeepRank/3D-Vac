#!/bin/bash
#SBATCH --partition thin
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=128
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/flattened_cnn_shuffled-%J.out
#removed -e /projects/0/einf2380/data/training_logs/I/flattened_cnn_shuffled-%J.err

## load modules
conda init bash
source activate deeprank
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/lib64'

srun python -u flatten_random_forest.py
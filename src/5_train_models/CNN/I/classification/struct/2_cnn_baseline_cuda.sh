#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --ntasks 10
#SBATCH --time 30:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/cnn_classification_struct_cuda-%J.out

## load modules
source activate deeprank
module load 2021
module load foss/2021a
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21,roundup_power2_divisions=4
# usage: srun python -u cnn_baseline.py <arguments for the script> --with-cuda

srun python -u cnn_baseline.py "$@" --with-cuda 

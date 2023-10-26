#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 08:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/performances_cnn-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/performances_cnn-%J.err

## load modules
conda init bash
source activate 3D-Vac
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/lib64'

srun python -u get_cnn_performances.py

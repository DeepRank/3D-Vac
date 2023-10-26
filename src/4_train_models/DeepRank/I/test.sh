#!/bin/bash
#SBATCH --partition thin
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32
#SBATCH --time 20:00:00

## load modules
conda init bash
source activate 3D-Vac
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/lib64'

srun python -u $1 --with-cuda 0 \
    --exp-name $2 \
    --model $3 \
    --data-path $4 \
    -E $5 --batch $6 \
    --output-dir /projects/0/einf2380/data/pMHCI/trained_models/CNN \

# srun python -u cnn_onefold_reg.py --with-cuda 0 --exp-name cpu_test --model CnnRegGroupConv --data-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled_subsample/shuffled/0 -E 2 --batch 256 --output-dir /projects/0/einf2380/data/pMHCI/trained_models/CNN
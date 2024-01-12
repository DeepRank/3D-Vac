#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 18
#SBATCH --time 8:00:00
#SBATCH --mem=120G

#removed -o /projects/0/einf2380/data/training_logs/II/cnn_classification_struct_cuda-%J.out
#removed -e /projects/0/einf2380/data/training_logs/II/cnn_classification_struct_cuda-%J.err
#removed --mem-per-gpu 39440

## load modules
source activate deeprank
#module load 2022
#module load OpenMPI/4.1.4-GCC-11.3.0
#module load 2021
#module load foss/2021a
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21,roundup_power2_divisions=4
# usage: sbatch src/5_train_models/CNN/II/classification/struct/2_cnn_baseline_cuda.sh

# srun python -u cnn_baseline_onefold.py --with-cuda 1 \
#     --exp-name hla_drb1_0101_15mers_4conv \
#     --model CnnClassification4Conv \
#     -E 20 -c $2 --batch 350 --task-id $1

srun python -u cnn_onefold_noPSSM.py --with-cuda 1 \
    --exp-name hla_drb1_0101_15mers_4conv_noPSSM \
    --model CnnClassification4Conv \
    -E 20 -c $2 --batch 350 --task-id $1

#nvidia-smi
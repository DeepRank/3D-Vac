#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 2
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 08:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/cnn_class_AllMers-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/cnn_class_AllMers-%J.err

## load modules
source activate 3D-Vac
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/lib64'

#module load 2022
#module load foss/2021a
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21,roundup_power2_divisions=4
# usage: sbatch src/5_train_models/CNN/II/classification/struct/2_cnn_baseline_cuda.sh

srun python -u cnn_onefold.py --with-cuda 2 \
    --exp-name clusterd_pepts_all_mers_Conv4 \
    --model CnnClassification4Conv \
    --data-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/all_HLA_quant_clustered_peptides_10fold/clustered/5/ \
    -E 10 --batch 128 \
    --output-dir /projects/0/einf2380/data/pMHCI/trained_models/CNN
#   --task-id $1
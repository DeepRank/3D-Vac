#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=18
#SBATCH --time 20:00:00
#removed -o /projects/0/einf2380/data/training_logs/I/cnn_class_AllMers-%J.out
#removed -e /projects/0/einf2380/data/training_logs/I/cnn_class_AllMers-%J.err

## load modules
conda init bash
source activate 3D-Vac
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/lib64'

srun python -u $1 --with-cuda 1 \
    --exp-name $2 \
    --model $3 \
    --data-path $4 \
    -E $5 --batch $6 \
    --output-dir /projects/0/einf2380/data/pMHCI/trained_models/CNN \

# srun python -u cnn_onefold.py --with-cuda 1 \
#     --exp-name clustered_alleles_noPSSM \
#     --model CnnClass4ConvKS3 \
#     --data-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_allele_clusters/clustered \
#     -E 10 --batch 256 \
#     --output-dir /projects/0/einf2380/data/pMHCI/trained_models/CNN \
#     --task-id 0

# srun python -u cnn_onefold_reg.py --with-cuda 1 \
#     --exp-name clustered_alleles_noPSSM \
#     --model CnnReg4ConvKS3 \
#     --data-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_allele_clusters/clustered \
#     -E 10 --batch 256 \
#     --output-dir /projects/0/einf2380/data/pMHCI/trained_models/CNN \
#     --task-id 0

# srun python -u cnn_onefold.py --with-cuda 1 \
#     --exp-name clusterd_pepts_all_mers_noPSSM \
#     --model CnnClass4ConvKS3 \
#     --data-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/all_HLA_quant_clustered_peptides_10fold/clustered \
#     -E 10 --batch 256 \
#     --output-dir /projects/0/einf2380/data/pMHCI/trained_models/CNN \
#     --task-id 3

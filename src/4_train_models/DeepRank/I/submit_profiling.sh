#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 08:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/I/profile_cnn-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/profile_cnn-%J.err

## load modules
conda init bash
source activate 3D-Vac
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/lib64'

srun python -u profile_network.py --with-cuda 1 \
    --exp-name profile_shuffled_old_data \
    --model CnnClassGroupConv \
    --data-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative_backup/splits85/clustered/0 \
    -E 3 --batch 256 \
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

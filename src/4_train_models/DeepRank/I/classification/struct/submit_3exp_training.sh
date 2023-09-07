#!/bin/bash
#removed --partition thin
#removed --array=0-9
#removed --gpus 1
#removed --nodes 1
#removed --ntasks 1
#removed --cpus-per-task 1
#removed --time 00:01:00
#removed -o /projects/0/einf2380/data/training_logs/II/cnn_classification_struct_cuda-%J.out
#removed -e /projects/0/einf2380/data/training_logs/II/cnn_classification_struct_cuda-%J.err
#removed --mem-per-gpu 39440

# conda init bash
# conda activate 3D-Vac

# #TEST
# sbatch -o /projects/0/einf2380/data/training_logs/I/test_cnn_regression_%J.out \
# -e /projects/0/einf2380/data/training_logs/I/test_cnn_regression_%J.err \
# --job-name=test_reg \
# train_single_fold.sh cnn_onefold_reg.py shuffled_test CnnRegGroupConv \
# /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled_subsample/shuffled/0 \
# 2 128

sbatch -o /projects/0/einf2380/data/training_logs/I/test_cnn_classification_%J.out \
-e /projects/0/einf2380/data/training_logs/I/test_cnn_classification_%J.err \
--job-name=test_class \
train_single_fold.sh cnn_onefold.py shuffled_test CnnClass4ConvKS3Lin128ChannExpand \
/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled_subsample/shuffled/0 \
2 128

# #REGRESSION
# ## Shuffled
# sbatch -o /projects/0/einf2380/data/training_logs/I/cnn_regression_shuffled_%J.out \
# -e /projects/0/einf2380/data/training_logs/I/cnn_regression_shuffled_%J.err \
# --job-name=shuff_reg \
# train_single_fold.sh cnn_onefold_reg.py shuffled_noPSSM_sumFeat_bn CnnRegGroupConv_4Conv \
# /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled/shuffled/0 \
# 15 128 \
# ## Peptide clustered
# sbatch -o /projects/0/einf2380/data/training_logs/I/cnn_regression_peptClust_%J.out \
# -e /projects/0/einf2380/data/training_logs/I/cnn_regression_peptClust_%J.err \
# --job-name=pept_reg \
# train_single_fold.sh cnn_onefold_reg.py clustPept_noPSSM_sumFeat_bn CnnRegGroupConv_4Conv \
# /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quantitative_gibbs_clust_10_3/clustered/0 \
# 15 128 \
# ## Allele clustered
# sbatch -o /projects/0/einf2380/data/training_logs/I/cnn_regression_alleleClust_%J.out \
# -e /projects/0/einf2380/data/training_logs/I/cnn_regression_alleleClust_%J.err \
# --job-name=alle_reg \
# train_single_fold.sh cnn_onefold_reg.py clustAllele_noPSSM_sumFeat_bn CnnRegGroupConv_4Conv \
# /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_allele_clusters/clustered/0 \
# 15 128 \

# # CLASSIFICATION
# ## Shuffled
# sbatch -o /projects/0/einf2380/data/training_logs/I/cnn_classification_shuffled_%J.out \
# -e /projects/0/einf2380/data/training_logs/I/cnn_classification_shuffled_%J.err \
# --job-name=shuff_class \
# train_single_fold.sh cnn_onefold.py shuffled_Cnn CnnClass4ConvKS3Lin128ChannExpand \
# /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled/shuffled/0 \
# 15 128 \
# # Peptide clustered
# sbatch -o /projects/0/einf2380/data/training_logs/I/cnn_classification_peptClust_%J.out \
# -e /projects/0/einf2380/data/training_logs/I/cnn_classification_peptClust_%J.err \
# --job-name=pept_class \
# train_single_fold.sh cnn_onefold.py clustPept_Cnn CnnClass4ConvKS3Lin128ChannExpand \
# /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quantitative_gibbs_clust_10_3/clustered/0 \
# 15 128 \
# ## Allele clustered
# sbatch -o /projects/0/einf2380/data/training_logs/I/cnn_classification_alleleClust_%J.out \
# -e /projects/0/einf2380/data/training_logs/I/cnn_classification_alleleClust_%J.err \
# --job-name=alle_class \
# train_single_fold.sh cnn_onefold.py clustAllele_Cnn CnnClass4ConvKS3Lin128ChannExpand \
# /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_allele_clusters/clustered/0 \
# 15 128 \

## Marieke's clustering
# sbatch -o /projects/0/einf2380/data/training_logs/I/cnn_classification_MarClust_%J.out \
# -e /projects/0/einf2380/data/training_logs/I/cnn_classification_MarClust_%J.err \
# --job-name=Marieke_class \
# train_single_fold.sh cnn_onefold.py clustMarieke_Cnn CnnClass4ConvKS3Lin128ChannExpand \
# /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/split_HLA_quant_Marieke_clust/clustered/3 \
# 15 128 \
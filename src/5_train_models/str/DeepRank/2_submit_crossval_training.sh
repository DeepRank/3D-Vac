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
source activate deeprank

# CLASSIFICATION
## Shuffled
EXPERIMENT=Shuffled
for FOLD_NUMBER in $(seq 1 5); do
    sbatch -o /projects/0/einf2380/data/training_logs/I/CNN_shuffled/cnn_class_CrossVal_shuffled_${FOLD_NUMBER}_%J.out \
    -e /projects/0/einf2380/data/training_logs/I/CNN_shuffled/cnn_class_CrossVal_shuffled_${FOLD_NUMBER}_%J.err \
    --job-name=CNN_shuff_class \
    train_single_fold.sh cnn_onefold.py shuffled_Cnn CnnClass4ConvKS3Lin128ChannExpand $FOLD_NUMBER \
    /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/CrossValidation/${EXPERIMENT}/ \
    20 128
done

## Allele clustered
EXPERIMENT=AlleleClustered
for FOLD_NUMBER in $(seq 1 5); do
    sbatch -o /projects/0/einf2380/data/training_logs/I/CNN_clustered/cnn_class_CrossVal_clustered_${FOLD_NUMBER}_%J.out \
    -e /projects/0/einf2380/data/training_logs/I/CNN_clustered/cnn_class_CrossVal_clustered_${FOLD_NUMBER}_%J.err \
    --job-name=CNN_alle_class \
    train_single_fold.sh cnn_onefold.py clustAllele_Cnn CnnClass4ConvKS3Lin128ChannExpand $FOLD_NUMBER \
    /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/CrossValidation/${EXPERIMENT}/ \
    20 128
done
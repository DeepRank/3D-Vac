#!/bin/bash
#SBATCH --partition thin
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/cnn_test_classification_struct_cpu-%J.out

source activate deeprank

# usage: srun python -u cnn_baseline.py <arguments for the script>
srun python -u cnn_performances.py -m /projects/0/einf2380/data/pMHCI/trained_models/CNN/clustAllele_noPSSM_sumFeat_bn/CnnClassGroupConv/best_valid_model.pth.tar \
-t /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_allele_clusters/clustered/0/test.hdf5 -o test

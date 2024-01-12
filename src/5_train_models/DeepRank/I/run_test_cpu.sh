#!/bin/bash

sbatch -o /projects/0/einf2380/data/training_logs/I/test_cnn_classification_%J.out \
-e /projects/0/einf2380/data/training_logs/I/test_cnn_classification_%J.err \
--job-name=test_class \
test.sh cnn_onefold.py cpu_test CnnClassGroupConv_4Conv \
/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/splits_HLA_quant_shuffled_subsample/shuffled/0 \
2 128
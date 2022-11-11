#!/bin/bash
#SBATCH --partition thin
#SBATCH --nodes 1
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 10
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/II/cnn_test_classification_struct_cpu-%J.out

source activate deeprank

module load 2021
module load foss/2021a

# usage: srun python -u cnn_baseline.py <arguments for the script>
srun python -u cnn_performances.py -C --splits-path /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_01_01_15-mers \
    --output-dir /projects/0/einf2380/data/pMHCII/trained_models/incrChannels/clustered/

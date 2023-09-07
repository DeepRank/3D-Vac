#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --partition=fat
#SBATCH --time=01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/II/generate_features.out

# Load modules:
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0

# Activate conda environment:
source activate deeprank
#alias TARGET_INPUT_CSV="/projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII.csv"

# Start the script:
srun python -u generate_features.py \
    --input-folder /projects/0/einf2380/data/pMHCII/features_input_folder/hla_drb1_0101_15mers \
    --h5out /projects/0/einf2380/data/pMHCII/features_output_folder/CNN/hla_drb1_0101_15mers/original/hla_drb1_0101_15mers.hdf5
#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 1
#SBATCH --time 10:00:00
# removed -o /projects/0/einf2380/data/training_logs/mlp_classification_seq_cuda-%J.out

## load modules
source activate deeprank
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0
# usage: srun python -u mlp_reg_baseline.py <arguments for the script>

srun python -u mlp_baseline.py -o test_MHCII_mlp -e sparse -f /home/dmarz/3D-Vac/data/external/processed/IDs_BA_DRB0101_MHCII_15mers.csv

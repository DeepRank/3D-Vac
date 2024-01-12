#!/bin/bash
#SBATCH --partition thin
#SBATCH --nodes 1
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 12
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/training_logs/mlp_classification_seq_cpu-%J.out

## load modules
source activate deeprank
module load 2020
module load OpenMPI/4.0.3-GCC-9.3.0
# usage: srun python -u mlp_reg_baseline.py <arguments for the script>

srun python -u mlp_baseline.py  -o test_MHCII_mlp -e sparse -f /projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv

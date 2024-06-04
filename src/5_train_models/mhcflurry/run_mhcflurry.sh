
#!/bin/bash

#SBATCH --nodes 1
#SBATCH --partition thin
#SBATCH --time 10:00:00
#SBATCH --ntasks 10
#SBATCH --cpus-per-task 2
#SBATCH -o /projects/0/einf2380/data/training_logs/I/MHCflurry/training_MHCflurry_shuffled_exp-%J.out
#SBATCH -e /projects/0/einf2380/data/training_logs/I/MHCflurry/training_MHCflurry_shuffled_exp-%J.err

python -u /home/dmarz/softwares/mhcflurry/mhcflurry/train_pan_allele_models_command.py \
    --data /projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_shuffled_train_validation.csv \
    --out-models-dir /projects/0/einf2380/data/pMHCI/trained_models/mhcflurry_rerun/shuffled \
    --hyperparameters /home/dmarz/3D-Vac/src/4_train_models/mhcflurry/hyperparameters.json \

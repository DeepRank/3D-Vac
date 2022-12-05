#!/bin/bash
#SBATCH --job-name align_pdbs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=thin
#SBATCH --time=01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db3/align_pdb_I-%J.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

export pdbs_path='/projects/0/einf2380/data/pMHCI/features_input_folder/exp_nmers_all_HLA_quantitative/pdb_renumbered/\*'
export original_pdb='/projects/0/einf2380/data/pMHCI/db2_selected_models_1/BA/68001_69000/BA-68085/pdb/BA-68085.pdb'
export alignment_template='/projects/0/einf2380/data/pMHCI/alignment/alignment_template.pdb'

#Copy the template file
cp $original_pdb $alignment_template
# Align all the models to the template file
srun --job-name first_align \
    python -u align_pdb.py --pdbs-path $pdbs_path --template $alignment_template --n-cores 128
Orient the template file on the aligned peptides PCA
srun --job-name orient_peptides --dependency=afterok:first_align \
      python -u orient_on_pept_PCA.py --pdbs-path $pdbs_path --template $alignment_template
# Align all the models to the re-oriented template file
srun --dependency=afterok:first_align:orient_peptides 
      python -u align_pdb.py --pdbs-path $pdbs_path --template $alignment_template --n-cores 128

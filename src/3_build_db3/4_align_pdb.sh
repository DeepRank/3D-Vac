#!/bin/bash
#SBATCH --job-name align_pdbs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=thin
#SBATCH --time=02:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db3/align_pdb_3step_I-%J.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

# export pdbs_path='/projects/0/einf2380/data/pMHCI/db2_selected_models_1/BA/'
export pdbs_path='/projects/0/einf2380/data/pMHCI/db2_selected_models/BA'
export original_pdb='/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/68001_69000/BA-68085/pdb/BA-68085.pdb'
export alignment_template='/projects/0/einf2380/data/pMHCI/alignment/alignment_template.pdb'

# Renumber the pdbs
echo "RENUMBER THE PDBS"
srun --job-name renumber python renumber_pdbs.py --n-cores 128 --folder $pdbs_path

#Copy the template file
echo "COPY TEMPLATE FILE"
cp $original_pdb $alignment_template

# Align all the models to the template file
echo "FIRST ALIGNMENT"
srun --dependency=afterok:renumber --job-name first_align python -u align_pdb.py --pdbs-path $pdbs_path --template $alignment_template --n-cores 128

# Orient all the pdbs on the aligned peptides PCA
echo "ORIENT PEPTIDES ON PCA"
srun --dependency=afterok:first_align --job-name peptide_pca python -u orient_on_pept_PCA.py --pdbs-path $pdbs_path --n-cores 128


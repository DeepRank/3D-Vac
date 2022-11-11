#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=thin
#SBATCH --time=05:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/alignpdbs_II.out

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

export pdbs_path='/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/\*/\*/pdb/\*.pdb'
export original_pdb='/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/55001_56000/BA_55613_2IAN/pdb/BA_55613.pdb'
export alignment_template='/projects/0/einf2380/data/pMHCII/3D_models/alignment/alignment_template.pdb'

#Copy the template file
echo "COPY TEMPLATE"
cp $original_pdb $alignment_template
# Align all the models to the template file
#srun --job-name first_align  \
echo "FIRST ALIGNMENT"
python -u align_pdb.py --pdbs-path $pdbs_path --template $alignment_template --n-cores 128

# Orient the template file on the aligned peptides PCA
#srun --job-name orient_peptides \
#    --dependency=afterany:first_align \
echo "ORIENT PEPTIDES ON PCA"
#TODO: make this script rotate all the models, so there is no need for the second alignment
python -u orient_on_pept_PCA.py --pdbs-path $pdbs_path --n-cores 128

# Align all the models to the re-oriented template file
#srun --dependency=afterany:first_align:orient_peptides \
#echo "SECOND ALIGNMENT"
#python -u align_pdb.py --pdbs-path $pdbs_path --template $alignment_template --n-cores 128
    

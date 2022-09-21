#!/bin/bash

#removed --nodes=1
#removed --ntasks-per-node=128
#removed --cpus-per-task=128
#removed --partition=thin
#removed --time=05:00:00

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

export pdbs_path='/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/\*/\*/pdb/\*.pdb'
export original_pdb='/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/55001_56000/BA_55613_2IAN/pdb/BA_55613.BL00200001.pdb'
export alignment_template='/projects/0/einf2380/data/pMHCII/3D_models/alignment/alignment_template.pdb'

#Copy the template file
cp $original_pdb $alignment_template
# Align all the models to the template file
#srun --job-name first_align  \
python -u align_pdb.py --pdbs-path $pdbs_path --template $alignment_template 

# Orient the template file on the aligned peptides PCA
#srun --job-name orient_peptides \
#    --dependency=afterany:first_align \
python -u orient_on_pept_PCA.py --pdbs-path $pdbs_path --template $alignment_template

# Align all the models to the re-oriented template file
#srun --dependency=afterany:first_align:orient_peptides \
python -u align_pdb.py --pdbs-path $pdbs_path --template $alignment_template  
    

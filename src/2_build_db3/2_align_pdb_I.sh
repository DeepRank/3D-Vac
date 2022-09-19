#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=thin
#SBATCH --time=01:00:00

# Load modules:
module load 2021
module load foss/2021a

# Activate conda env:
source activate deeprank

export pdbs_path='/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/*/*/pdb/*.pdb'
export original_pdb='/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/55001_56000/BA_55224_3MRC/pdb/BA-55224.BL00160001.pdb'
export alignment_template='/projects/0/einf2380/data/pMHCI/alingment/alignment_template.pdb'

#Copy the template file
cp $original_pdb $alignment_template
# Align all the models to the template file
srun --job-name first_align \
    python -u align_pdb.py --pdbs-path $pdbs_path --template $alignment_template
# Orient the template file on the aligned peptides PCA
srun --job-name orient_peptides \
    --dependency=afterany:first_align \ 
    python -u orient_on_pept_PCA.py -pdbs_path $pdbs_path \ 
    -template $alignment_template
# Align all the models to the re-oriented template file
srun --dependency=afterany:first_align:orient_peptides \ 
    python -u align_pdb.py --pdbs-path $pdbs_path \ 
    --template $alignment_template

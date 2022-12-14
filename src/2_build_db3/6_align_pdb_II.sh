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

export pdbs_path='/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/'
export original_pdb='/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/55001_56000/BA_55613_2IAN/pdb/BA_55613.pdb'
export alignment_template='/projects/0/einf2380/data/pMHCII/3D_models/alignment/alignment_template.pdb'

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
srun --dependency=afterok:first_align python -u orient_on_pept_PCA.py --pdbs-path $pdbs_path --n-cores 128

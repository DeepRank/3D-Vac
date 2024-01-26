#!/bin/bash

# Activate conda env:
source activate deeprank

python build_pssm_input.py --output-folder /projects/0/einf2380/data/pMHCII/pssm_raw/all_mhc --input-file-db1 /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII_noduplicates.csv
python create_raw_pssm.py --workdir /projects/0/einf2380/data/pMHCII/pssm_raw/all_mhc --blast-db /projects/0/einf2380/data/blast_dbs/all_mhc/all_mhc
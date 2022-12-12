#!/bin/bash

# Activate conda env:
source activate deeprank

python build_pssm_input.py --output-folder  --input-file-db1 
python create_raw_pssm.py --workdir /projects/0/einf2380/data/pMHCII/pssm_raw/hla_drb1_0101
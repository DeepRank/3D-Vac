#!/bin/bash

python -u build_db2.py --input-csv /projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative_only_eq.csv  \
    --models-dir /projects/0/einf2380/data/pMHCI/3d_models/BA/\*/\* \
    --mhc-class I \
    --num-nodes 10
    
# Inputs: generated db1 in `data/external/processed`.
# Output: models in the `models` folder.
# Run `python src/2_build_db2/build_db2.py --help` for more details on how the script works.
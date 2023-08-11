#!/bin/bash

python -u build_db2.py --input-csv /projects/0/einf2380/data/external/processed/I/Full_pMHCI.csv  \
    --models-dir /projects/0/einf2380/data/pMHCI/3d_models_production/ALL/\*/\* \
    --mhc-class I \
    --num-nodes 10
    
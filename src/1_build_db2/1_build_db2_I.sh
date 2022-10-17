#!/bin/bash

python -u build_db2.py --input-csv /projects/0/einf2380/data/external/processed/I/BA_pMHCI.csv  \
    --models-dir /projects/0/einf2380/data/pMHCI/3D_models/BA/\*/\* \
    --mhc-class I \
    --num-nodes 10
    
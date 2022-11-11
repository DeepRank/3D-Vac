#!/bin/bash
source activate deeprank

python -u build_db2.py --input-csv /projects/0/einf2380/data/external/processed/II/IDs_BA_MHCII_noduplicates.csv \
    --models-dir /projects/0/einf2380/data/pMHCII/3D_models/BA/\*/\* \
    --mhc-class II \
    --num-nodes 10

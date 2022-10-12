#!/bin/bash

source activate deeprank

python -u build_db2.py --input-csv /projects/0/einf2380/data/external/processed/II/BA_pMHCII.csv \
    --models-dir \
    --mhc-class II \
    --num-nodes 10

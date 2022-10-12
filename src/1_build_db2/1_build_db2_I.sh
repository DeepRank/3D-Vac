#!/bin/bash

conda activate deeprank

python -u build_db2.py --input-csv /home/severin/3D-Vac/data/external/processed/I/BA_pMHCI_mock499061.csv \
    --models-dir /projects/0/einf2380/data/pMHCI/models/BA_4/\*/\* \
    --mhc-class I \
    --num-nodes 2
    
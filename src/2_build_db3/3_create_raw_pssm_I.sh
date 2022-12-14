#!/bin/bash
#SBATCH --job-name create_raw_pssm
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db3/build_raw_pssm_I-%J.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 32
#SBATCH --time 03:00:00
# Activate conda env:

source activate deeprank

python build_pssm_input.py --output-folder /projects/0/einf2380/data/pMHCI/pssm_raw/all_mhc --input-file-db1 /projects/0/einf2380/data/external/processed/I/BA_ids_pMHCI_human_nonhuman.csv
python create_raw_pssm.py --workdir /projects/0/einf2380/data/pMHCI/pssm_raw/all_mhc --blast-db /projects/0/einf2380/data/blast_dbs/all_mhc/all_mhc
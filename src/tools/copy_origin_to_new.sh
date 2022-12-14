#!/bin/bash
#SBATCH -p thin
#SBATCH  --cpus-per-task 64
#SBATCH --job-name copy_orgs
#SBATCH --time 00:30:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db3/copy_orgs-%J.out

source activate deeprank

python copy_origin_to_new.py --models-dir /projects/0/einf2380/data/pMHCI/db2_selected_models_1/BA
#!/bin/bash
#SBATCH -p thin
#SBATCH -n 1 
#SBATCH --cpus-per-task 128
#SBATCH --job-name renumber_pdbs
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db3/renumber-pdbs-%J.out

# Load conda environment
source activate deeprank

# cp -a /projects/0/einf2380/data/pMHCI/features_input_folder/exp_nmers_all_HLA_quantitative/pdb/. /projects/0/einf2380/data/pMHCI/features_input_folder/exp_nmers_all_HLA_quantitative/pdb_renumbered
python renumber_pdbs.py --n-cores 128 --folder /projects/0/einf2380/data/pMHCI/features_input_folder/exp_nmers_all_HLA_quantitative/pdb/
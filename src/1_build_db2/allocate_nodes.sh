#!/bin/bash
#SBATCH --job-name allocate_nodes
#SBATCH -o /projects/0/einf2380/data/modelling_logs/MHCI_allocate_nodes.out
#SBATCH -n 1
#SBATCH -c 1

python -u allocate_nodes.py "$@"
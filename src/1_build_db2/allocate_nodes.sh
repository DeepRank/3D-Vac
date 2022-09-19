#!/bin/bash
#SBATCH --job-name allocate_nodes
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/allocate_nodes-%J.out
#SBATCH -n 1
#SBATCH -c 1

conda init
source activate deeprank

python -u allocate_nodes.py "$@"
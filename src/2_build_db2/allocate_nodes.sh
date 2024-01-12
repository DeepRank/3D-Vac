#!/bin/bash
#SBATCH --job-name allocate_nodes
#SBATCH -n 1
#SBATCH -c 1

source activate deeprank

python -u allocate_nodes.py "$@"
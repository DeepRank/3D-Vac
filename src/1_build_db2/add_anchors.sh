#!/bin/bash

#SBATCH -p thin
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -o /projects/0/einf2380/data/modelling_logs/I/db2/add_anchor_column-%J.out

source activate deeprank

# usage: python add_anchors <arguments>
python add_anchors.py "$@"

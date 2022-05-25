#!/bin/bash
#SBATCH -p thin
#SBATCH --nodes 1
#SBATCH --time 01:00:00
#SBATCH -o /projects/0/einf2380/data/modelling_logs/job-%J.out
#SBATCH -p thin
#SBTACH --exclusive
#SBATCH --export ALL
#SBATCH -n 1 -c 128

## load modules

## usage: python ./run_wrapper.py [number of cores] [start line] [end line] 
python ./run_wrapper.py 128 $1 $2

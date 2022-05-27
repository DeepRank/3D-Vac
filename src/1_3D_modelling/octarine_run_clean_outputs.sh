#!/bin/bash
#$ -l h_rt=3:00:00
#$ -cwd
#$ -V 
#$ -p -5
#$ -o $HOME/3d-epipred/modelling/logs/
#$ -e $HOME/3d-epipred/modelling/logs/

## Usage example without array: qsub -q all.q@narrativum.umcn.nl octarine_run_clean_outputs.sh
mpiexec -n 32 -host localhost python ./clean_outputs.py

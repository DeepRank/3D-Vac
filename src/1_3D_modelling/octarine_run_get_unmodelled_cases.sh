#!/bin/bash
#$ -l h_rt=1:00:00
#$ -cwd
#$ -V 
#$ -o $HOME/3d-epipred/modelling/logs/
#$ -e $HOME/3d-epipred/modelling/logs/

## Usage example without array: qsub -q all.q@narrativum.umcn.nl octarine_run_get_unmodelled_cases.sh
python ./get_unmodelled_cases.py

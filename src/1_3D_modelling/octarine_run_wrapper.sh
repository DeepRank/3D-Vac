#!/bin/bash
#$ -l h_rt=50:00:00
#$ -cwd
#$ -V 
#$ -p -5
#$ -o $HOME/3d-epipred/modelling/logs/
#$ -e $HOME/3d-epipred/modelling/logs/

## Usage example without array: qsub -q all.q@narrativum.umcn.nl octarine_run_wrapper.sh
## Args: Number of cores, start_row, end_row (prev batch dimension, batch id)
python ./run_wrapper.py $1 $2 $3

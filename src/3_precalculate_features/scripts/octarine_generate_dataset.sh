#!/bin/bash
#$ -l h_rt=03:00:00
#$ -cwd
#$ -V

##Usage: qsub -q all.q@narrativum.umcn.nl  octarine_generate_dataset.sh
mpiexec -n 4 -host localhost python3 ./generate_dataset.py

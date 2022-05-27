#!/bin/bash
#$ -l h_rt=00:10:00
#$ -cwd
#$ -V

##Usage: qsub -q all.q@narrativum.umcn.nl  test_mpi.sh
mpiexec -n 8 -host localhost python ./test_mpi.py

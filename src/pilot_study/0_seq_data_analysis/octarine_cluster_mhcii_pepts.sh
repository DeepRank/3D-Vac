#!/bin/bash
#$ -l h_rt=03:00:00
#$ -cwd
#$ -V

##Usage: qsub -pe smp n_cores -q all.q@narrativum.umcn.nl octarine_cluster_mhcii_pepts.sh n_cores
python ./cluster_mhcii_pepts_per_allele.py $1

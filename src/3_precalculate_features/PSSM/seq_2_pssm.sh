#!/usr/bin/env bash
# Dario Marzella
# 20-apr-2020 13:21

#generate pssm from alignment
# /home/dariom/ncbi-blast-2.11.0+/bin/psiblast -query $1 -num_iterations 2 -out_ascii_pssm $2 -db /home/dariom/3d-epipred/CNN/PSSM/all_mhc_db/all_mhc_db
/home/dariom/ncbi-blast-2.11.0+/bin/psiblast -query $1 -num_iterations 2 -out_ascii_pssm $2 -db /home/dariom/3d-epipred/CNN/PSSM/hla_db/hla_db

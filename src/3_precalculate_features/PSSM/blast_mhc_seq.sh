#!/usr/bin/env bash

#Usage blast_mhc_seq.sh <fasta_infile> <outfiles_name>
#Extensions .out and .pssm will be added to <outfile_names>
/home/dariom/ncbi-blast-2.11.0+/bin/psiblast -query $1 -db /home/dariom/3d-epipred/CNN/PSSM/uniprot_mhc_db/uniprot_mhc_db -out $2.out -out_ascii_pssm $2.pssm

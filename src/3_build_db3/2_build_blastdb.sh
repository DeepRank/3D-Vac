#!/bin/bash
# usage: makeblast -in <fasta file of every human hla proteins> -dbtype prot
export db_path="/projects/0/einf2380/data/blast_dbs/all_mhc"
makeblastdb -in $db_path/all_mhc_prot.fasta -dbtype prot -out $db_path/all_mhc
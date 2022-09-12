#!/bin/bash
# usage: makeblast -in <fasta file of every human hla proteins> -dbtype prot
makeblastdb -in ../../data/pssm/blast_dbs/hla_prot.fasta -dbtype prot
import sys
from Bio import SeqIO
from Bio.PDB import PDBParser
import glob
import os
import warnings
import math

warnings.filterwarnings("ignore");

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

pdb_path = "/projects/0/einf2380/data/pMHCI/models/temp/**/*.BL00020001.pdb"
PDBfiles = glob.glob(pdb_path) # the list of PDB files are build on the basis of pdb_path glob constructor
print(f"Are pdb files unique: {len(set(PDBfiles)) == len(PDBfiles)}");
print(f"number of fasta elements: {len(PDBfiles)}")
fastas = "" # retreive the fastas structures from the PDB files
f_rows = {};

for PDBfile in PDBfiles:
    if os.stat(PDBfile).st_size != 0:
        with open(PDBfile, "r") as pdb_f:
            pdb = PDBParser(QUIET=True).get_structure(PDBfile, PDBfile)
            for model in pdb:
                for chain in model:
                    seq = [];
                    if len(chain)>9:
                        for residue in chain:
                            seq.append(d3to1[residue.resname])
                        f_rows[PDBfile] = seq
for id,value in f_rows.items():
    fastas+= f">{id}\n {''.join(value)} \n"

with open("./targets.fasta", "w") as fasta_f:
    fasta_f.write(fastas)
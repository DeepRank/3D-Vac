#!/usr/bin/env python
# Dario Marzella
# 10-Ago-2021 19:00


from pssmgen import PSSM
import sys

pdb_dir = sys.argv[1]

# initiate the PSSM object
gen = PSSM('./')

# map PSSM and PDB to get consisitent files
gen.map_pssm(pssm_dir='pssm_raw', pdb_dir=pdb_dir, out_dir='pssm', chain = ['M'])

# write consistent files and move
gen.get_mapped_pdb()

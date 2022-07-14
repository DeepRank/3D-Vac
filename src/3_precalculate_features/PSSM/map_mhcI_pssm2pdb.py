#!/usr/bin/env python
# Dario Marzella
# 10-Ago-2021 19:00
from pssmgen import PSSM
from mpi4py import MPI
import os
import glob

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

folders = glob.glob("/projects/0/einf2380/data/pMHCI/pssm_mapped/BA/*/*")
step = int(len(folders)/size)
start = int(rank*step)
end = int((rank+1)*step)

if rank == 0:
    print(size)
if rank != size-1:
    cut_folders = folders[start:end]
else:
    cut_folders = folders[start:]


for wd in cut_folders:
    if not glob.glob(f"{wd}/pssm/*.pssm"): #generate the mapping only if the mapping is not generated already
        gen = PSSM(work_dir=wd)
        gen.map_pssm(pssm_dir='pssm_raw', pdb_dir="pdb", out_dir='pssm', chain = ['M'])
        # write consistent files and move
        gen.get_mapped_pdb()

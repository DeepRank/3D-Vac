import glob
import os
import argparse
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

arg_parser = argparse.ArgumentParser(
    description="""
    Use this script to create symlinks in the output folder (--output-folder) for pdb and pssm folders for db2. 
    This folder will be used by generate_features.py as input.
    """
)
arg_parser.add_argument("--output-folder", "-o",
    help="""
    Name of the folder in "/projects/0/einf2380/data/pMHCI/features_input_folder" to put the symlinks in.
    """,
    default="hla_02_01_9_mers"
)

a = arg_parser.parse_args()

input_folder = f"/projects/0/einf2380/data/pMHCI/features_input_folder/{a.output_folder}"
pdb_folder = f"{input_folder}/pdb"
pssm_folder = f"{input_folder}/pssm"

if rank == 0:
    try:
        os.makedirs(pdb_folder)
        os.makedirs(pssm_folder)
    except:
        pass

    print("globing pdb_files")
    pdb_files = np.array(glob.glob('/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/*/*/pdb/*.pdb'))
    print("globing pssm_files")
    pssm_files = np.array(glob.glob('/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/*/*/pssm/*.pssm'))

    print(f"pdb_files len: {pdb_files.shape[0]}")
    print(f"pssm_files len: {pssm_files.shape[0]}")

    pdb_files = np.array_split(pdb_files, size)
    pssm_files = np.array_split(pssm_files, size)

else:
    pdb_files = None
    pssm_files = None

pdb_files = comm.scatter(pdb_files, root=0)
pssm_files = comm.scatter(pssm_files, root=0)

for pdb in pdb_files:
        pdb_file_name = pdb.split("/")[-1]
        dest = f"{pdb_folder}/{pdb_file_name}"
        try:
            os.symlink(pdb, dest)
        except:
            pass

for pssm in pssm_files:
        pssm_file_name = pssm.split("/")[-1]
        dest = f"{pssm_folder}/{pssm_file_name}"
        try:
            os.symlink(pssm, dest)
        except:
            pass
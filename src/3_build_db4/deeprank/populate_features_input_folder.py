import glob
import os
import argparse
from mpi4py import MPI
import numpy as np
import subprocess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

arg_parser = argparse.ArgumentParser(
    description="""
    Use this script to create symlinks in the output folder (--output-folder) for pdb and pssm folders for db2. 
    This folder will be used by generate_features.py as input.
    """
)
arg_parser.add_argument("--input-folder", "-i",
    help="""
    Path to the input folder
    """,
    default="/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/*/*"
)
arg_parser.add_argument("--output-folder", "-o",
    help="""
    Path to the output folder
    """,
    default="/projects/0/einf2380/data/pMHCI/features_input_folder/hla_02_01_9_mers"
)

a = arg_parser.parse_args()

pdb_folder = f"{a.output_folder}/pdb"
pssm_folder = f"{a.output_folder}/pssm"

if rank == 0:
    try:
        os.makedirs(pdb_folder)
        os.makedirs(pssm_folder)
    except:
        pass

    print("globing pdb_files")
    pdb_files = np.array(glob.glob(f'{a.input_folder}/pdb/*.pdb'))
    print("globing pssm_files")
    pssm_files = [x for x in glob.glob(f'{a.input_folder}/pssm/*.pssm') if not x.endswith('.N.pdb.pssm')]
    pssm_files = np.array(pssm_files)


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
    if os.path.exists(dest):
        try:
            subprocess.check_call(f'rm {dest}', shell=True)
        except:
            print(f'Something went removing old symlink for pdb {pdb}')
    try: #Try to remove pre-existing symlink
        subprocess.check_call(f'rm {dest}')
    except:
        pass
    try: #Make new symlink
        os.symlink(pdb, dest)
    except:
        print(f'Something went wrong symlinking pdb {pdb}')

for pssm in pssm_files:
    pssm_file_name = pssm.split("/")[-1]
    dest = f"{pssm_folder}/{pssm_file_name}"
    if os.path.exists(dest):
        try:
            subprocess.check_call(f'rm {dest}', shell=True)
        except:
            print(f'Something went removing old symlink for pssm {pdb}')
    try: #Try to remove pre-existing symlink
        subprocess.check_call(f'rm {dest}')
    except:
        pass
    try: #Make new symlink
        os.symlink(pssm, dest)
    except:
        print(f'Something went wrong symlinking pssm {pssm}')
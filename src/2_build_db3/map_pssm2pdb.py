import argparse
import pandas as pd
import glob
from pathlib import Path
import numpy as np
from pssmgen import PSSM
import os
from mpi4py import MPI
import subprocess

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

arg_parser = argparse.ArgumentParser(
    description = """
    Map the generated raw PSSM to the PDB target structures. 
    """
)
arg_parser.add_argument("--csv-file", "-i",
    help="Name of db1 in data/external/processed/. Default BA_pMHCI.csv",
    default="../../data/external/processed/BA_pMHCI.csv"
)
arg_parser.add_argument("--alphachain-pssm", "-M",
    help="""
    MHC class alpha chain (M chain) raw pssm path
    """,
    required=True
)
arg_parser.add_argument("--betachain-pssm", "-N",
    help="""
    MHC class beta chain (N chain) raw pssm path
    """,
)

arg_parser.add_argument("--mhc-class", "-c",
    help="""
    MHC class
    """,
    default="I",
    choices=["I", "II"],
)

a = arg_parser.parse_args()

csv_path = f"{a.csv_file}"
df = pd.read_csv(csv_path)

if rank == 0:
    all_models = glob.glob(f"/projects/0/einf2380/data/pMHC{a.mhc_class}/db2_selected_models/BA/*/*/pdb/*.pdb")
    db2 = np.array([model for model in all_models if Path(model).stem.split(".")[0] in df["ID"].tolist()])
    db2 = np.array_split(db2, size)
else:
    db2 = None

db2 = comm.scatter(db2, root=0)

if a.mhc_class == 'I':
    chains = {'M':a.alphachain_pssm}
elif a.mhc_class == 'II':
    chains = {'M':a.alphachain_pssm, 'N':a.betachain_pssm}

for case in db2:
    work_dir = "/".join(case.split("/")[:-2])
    try:
        subprocess.check_call(f'mkdir {work_dir}/pssm_raw', shell=True)
    except:
        print(f'Warning: could not make pssm_raw folder for case {case}')
    try:
        subprocess.check_call(f'mkdir {work_dir}/pssm', shell=True)
    except:
        print(f'Warning: could not make pssm folder for case {case}')
    model_name = case.split("/")[-1].split('.')[0]
    for chain_id, path in chains.items():
        subprocess.check_call(f'cp {path} {work_dir}/pssm_raw/{model_name}.{chain_id}.pssm', shell=True)

    gen = PSSM(work_dir=work_dir)
    gen.map_pssm(pssm_dir="pssm_raw", pdb_dir="pdb", out_dir="pssm", chain=list(chains.keys()))

print(f"Finished mapping {db2.shape[0]} PSSM to PDB on {rank}")
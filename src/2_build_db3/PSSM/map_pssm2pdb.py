import argparse
import pandas as pd
import glob
from pathlib import Path
import numpy as np
from pssmgen import PSSM
import os
from mpi4py import MPI

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
    default="BA_pMHCI.csv"
)

a = arg_parser.parse_args()

csv_path = f"../../../data/external/processed/{a.csv_file}"
df = pd.read_csv(csv_path)

if rank == 0:
    all_models = glob.glob("/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/*/*/pdb/*.pdb")
    db2 = np.array([model for model in all_models if Path(model).stem.replace("-", "_").split(".")[0] in df["ID"].tolist()])
    db2 = np.array_split(db2, size)
else:
    db2 = None

db2 = comm.scatter(db2, root=0)

for case in db2:
        work_dir = "/".join(case.split("/")[:-2])
        gen = PSSM(work_dir=work_dir)
        gen.map_pssm(pssm_dir="pssm_raw", pdb_dir="pdb", out_dir="pssm", chain="M")

print(f"Finished mapping {db2.shape[0]} PSSM to PDB on {rank}")
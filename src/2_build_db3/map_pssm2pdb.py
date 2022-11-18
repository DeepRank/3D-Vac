import argparse
import pandas as pd
import glob
from pathlib import Path
import numpy as np
from pssmgen import PSSM
import os
import traceback
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
    MHC class alpha chain (M chain) raw pssm path containing all mhc alleles
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
df['ID'] = df['ID'].apply(lambda x: '-'.join(x.split('_')))

if rank == 0:
    all_models = []
    all_models_first = glob.glob(f"/projects/0/einf2380/data/pMHC{a.mhc_class}/db2_selected_models_1/BA/*")
    for folder in all_models_first:
        all_models.extend(glob.glob(os.path.join(folder, '*/pdb/*.pdb')))
    db2 = np.array_split(all_models, size)
else:
    db2 = None

db2 = comm.scatter(db2, root=0)

for case in db2:
    work_dir = "/".join(case.split("/")[:-2])
    # check which allele belongs to case
    allele = df[df['ID'] == Path(case).stem.split(".")[0]]['allele'].values[0]
    try:
        if a.mhc_class == 'I':
            chain_M =  glob.glob(os.path.join(a.alphachain_pssm, f'pssm_raw/*{allele}*.pssm'))[0]
            chains = {'M': chain_M}
            
        elif a.mhc_class == 'II':
                chain_M = glob.glob(path.join(a.alphachain_pssm, f'pssm_raw/*dra*{allele}*.pssm'))[0]
                chain_N = glob.glob(path.join(a.betachain_pssm, f'pssm_raw/*drb1*{allele}*.pssm'))[0]
                chains = {'M': chain_M, 'N': chain_N}
    except IndexError as ie:
        print(f'Could not find allele of case {case} in pssm db\n{ie}\n{traceback.format_exc()}')
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

print(f"Finished mapping {len(db2)} PSSM to PDB on {rank}")
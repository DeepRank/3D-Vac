# this script should be runned after map_pssm2pdb.py
import glob
from pdb2sql import pdb2sql
from mpi4py import MPI
import pandas as pd
import argparse
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

arg_parser = argparse.ArgumentParser(
    description="""
    This script takes peptides from `peptide` column of db1 (provided with --input-csv parameter) and encode it
    using one-hot encoding to generate a pseudo-PSSM of the peptide.
    The pseudo-PSSM is written into the pssm folder as BA_xyz.P.pssm.
    """
)
arg_parser.add_argument("--input-csv", "-i",
    help="Name of db1 in data/external/processed/. Default BA_pMHCI.csv.",
    default="../../data/external/processed/BA_pMHCI.csv",
)
arg_parser.add_argument("--mhc-class", "-m",
    help="""
    MHC class
    """,
    default="I",
    choices=["I", "II"],
)

a = arg_parser.parse_args()

pssm_folders = glob.glob(f"/projects/0/einf2380/data/pMHC{a.mhc_class}/db2_selected_models/BA/*/*")
pssm_template_path = "/projects/0/einf2380/data/templates/M_chain_mapped_template.pssm"

# make the peptide_sequences
df = pd.read_csv(f"{a.input_csv}")

# retrieve the first row of the pssm_template to make the template for the pseudo-PSSM
pssm_template = []
with open(pssm_template_path) as template_f:
    rows = [row.replace("\n", "").split() for row in template_f]
    pssm_template = rows[0]

if rank == 0:
    IDs = np.array(list(range(len(df))))
    IDs = np.array_split(IDs, size)
else:
    IDs = None

IDs = comm.scatter(IDs, root=0)

for idx in IDs:
    #if idx == 0:
    sequence = df.loc[idx,"peptide"]
    sequence_id = df.loc[idx, "ID"]
    peptide_pssm_rows = [pssm_template]
    for i,res in enumerate(sequence):
        pdbresi = str(i+1)
        pdbresn = res
        seqresi = pdbresi
        seqresn = pdbresn
        peptide_pssm_row = [pdbresi,pdbresn,seqresi,seqresn,*[str(0)]*21]
        onehot_pos = pssm_template.index(res.strip())
        peptide_pssm_row[onehot_pos] = str(1)
        peptide_pssm_rows.append(peptide_pssm_row) 
    #write the file
    peptide_pssm_path = [path for path in pssm_folders if sequence_id in path.split("/")[-1]][0] + "/pssm"
    peptide_pssm_file = glob.glob(f"{peptide_pssm_path}/*.M.pdb.pssm")[0].split("/")[-1].replace("M","P")
    peptide_pssm_complete_path = f"{peptide_pssm_path}/{peptide_pssm_file}"
    print(peptide_pssm_complete_path)
    to_write= "\n".join(["\t".join(row) for row in peptide_pssm_rows])
    with open(peptide_pssm_complete_path, "wb") as peptide_f:
        to_write = to_write.encode("utf8").strip()
        peptide_f.write(to_write)
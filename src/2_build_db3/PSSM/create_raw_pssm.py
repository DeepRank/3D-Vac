from pssmgen import PSSM
import argparse
from mpi4py import MPI
import pandas as pd
import glob
import numpy as np
import os

arg_parser = argparse.ArgumentParser(
    description="This script generates the unmapped pssm for each case and dumps it into the pssm_raw folder in the \
    db3_inputs folder."
)
arg_parser.add_argument("--input-csv", "-i",
    help="db1 name in data/external/processed to generate the raw PSSM for.",
    default="BA_pMHCI.csv",
)
arg_parser.add_argument("--psiblast-path", "-p",
    help="Path to psiblast executable. Default '/home/lepikhovd/softwares/blast/bin/psiblast'.",
    default='/home/lepikhovd/softwares/blast/bin/psiblast'
)
a = arg_parser.parse_args()

df = pd.read_csv(f"../../../data/external/processed/{a.input_csv}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    best_models = glob.glob("/projects/0/einf2380/data/pMHCI/db2_selected_models/BA/*/*") 
    db2 = np.array([model for model in best_models if "_".join(model.split("/")[-1].split("_")[0:-1]) in df["ID"].tolist()])
    db2 = np.array_split(db2, size)
else:
    db2 = None
db2 = comm.scatter(db2, root=0)

for case in db2:
        try: # create necessary directories for pssm
            os.mkdir(f"{case}/fasta")
        except:
            pass

        # initiate the PSSM object:
        gen = PSSM(work_dir=case)

        # get the fasta from pdb:
        gen.get_fasta(pdb_dir="pdb", chain=("M"), out_dir="fasta")

        # the package requires to rename the generated fasta fi

        # set psiblast executable, database and other psiblast parameters (here shows the defaults)
        gen.configure(blast_exe=a.psiblast_path,
                    database='../../../data/pssm/blast_dbs/hla_prot.fasta',
                    num_threads = 3, evalue=0.0001, comp_based_stats='T',
                    max_target_seqs=2000, num_iterations=3, outfmt=7,
                    save_each_pssm=True, save_pssm_after_last_round=True)

        # generates raw PSSM files by running BLAST with fasta files
        gen.get_pssm(fasta_dir='fasta', out_dir='pssm_raw', run=True, save_all_psiblast_output=True)
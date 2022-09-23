from pssmgen import PSSM
import argparse
from mpi4py import MPI
import pandas as pd
import glob
import numpy as np
import os
from Bio import SeqIO

arg_parser = argparse.ArgumentParser(
    description="This script generates the unmapped pssm for each case and dumps it into the pssm_raw folder in the \
    db3_inputs folder."
)
arg_parser.add_argument("--input-csv", "-i",
    help="db1 name in data/external/processed to generate the raw PSSM for.",
    default="../../data/external/processed/BA_pMHCI.csv",
)
arg_parser.add_argument("--psiblast-path", "-p",
    help="Path to psiblast executable., Necessary if psiblast is not callable by terminal",
)
arg_parser.add_argument("--mhc-class", "-m",
    help="""
    MHC class
    """,
    default="I",
    choices=["I", "II"],
)
a = arg_parser.parse_args()

df = pd.read_csv(f"{a.input_csv}")

if a.psiblast_path:
    psiblast = a.psiblast_path
else:
    psiblast = 'psiblast'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    best_models = glob.glob(f"/projects/0/einf2380/data/pMHC{a.mhc_class}/db2_selected_models/BA/*/*/pdb/*") 
    db2 = np.array([model for model in best_models if model.split("/")[-1].split(".")[0] in df["ID"].tolist()])
    db2 = np.array_split(db2, size)
else:
    db2 = None

db2 = comm.scatter(db2, root=0)
print(f'RANK: {rank}')
print(f'CASES: {db2}')

for model in db2:

    model_id = model.split('/')[-1].replace(".pdb", "")
    case = ('/').join(model.split('/')[:-2])

    try: # create necessary directories for pssm
        os.mkdir(f"{case}/fasta")
    except:
        pass

    with open(model, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            chain = record.id.split(':')[1]
            if chain != 'P':
                with open(f"{case}/fasta/{model_id}.{chain}.fasta", 'w') as outfasta:
                    outfasta.write('>' + record.id.replace('????', model_id) + '\n')
                    outfasta.write(str(record.seq) + '\n')

    # initiate the PSSM object:
    gen = PSSM(work_dir=case)

    # set psiblast executable, database and other psiblast parameters (here shows the defaults)
    gen.configure(blast_exe=psiblast,
                database='/projects/0/einf2380/data/blast_dbs/all_hla/all_hla',
                num_threads = 3, evalue=0.0001, comp_based_stats='T',
                max_target_seqs=2000, num_iterations=3, outfmt=7,
                save_each_pssm=True, save_pssm_after_last_round=True)

    # generates raw PSSM files by running BLAST with fasta files
    gen.get_pssm(fasta_dir='fasta', out_dir='pssm_raw', run=True, save_all_psiblast_output=True)
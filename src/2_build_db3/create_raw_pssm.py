from pssmgen import PSSM
import argparse
from mpi4py import MPI
import pandas as pd

arg_parser = argparse.ArgumentParser(
    description="This script generates the unmapped pssm for each case and dumps it into the pssm_raw folder in the \
    db3_inputs folder."
)
arg_parser.add_argument("--input-csv", "-i",
    help="db1 name in data/external/processed to generate the raw PSSM for.",
    default="BA_pMHCI.csv",
)
a = arg_parser.parse_args()

df = pd.read_csv(f"../../data/external/processed/{a.input_csv}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

batch = int(len(df)/size)
if rank == size-1:
    cases = len(df) - rank*batch
else:
    cases = batch

# create the raw_pssm, for the proof of principle work with the HLA-A*02:01 allele, only 1 sequence is required:

gen = PSSM(work_dir='/projects/0/einf2380/data/pMHCI/pssm_raw/hla_a_02_01')

# set psiblast executable, database and other psiblast parameters (here shows the defaults)
gen.configure(blast_exe='/home/lepikhovd/softwares/blast/bin/psiblast',
            database='/projects/0/einf2380/data/pMHCI/pssm_raw/hla_a_02_01/Human_MHC_data.fasta',
            num_threads = 3, evalue=0.0001, comp_based_stats='T',
            max_target_seqs=2000, num_iterations=3, outfmt=7,
            save_each_pssm=True, save_pssm_after_last_round=True)

# generates raw PSSM files by running BLAST with fasta files
gen.get_pssm(fasta_dir='fasta', out_dir='pssm_raw', run=True, save_all_psiblast_output=True)
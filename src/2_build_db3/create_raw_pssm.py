from pssmgen import PSSM
import argparse

arg_parser = argparse.ArgumentParser(
    description="Generate one raw pssm in the given folder. Requires a blast database and a fasta file. \
        The fasta file can be generated with src/tools/pdb_to_fasta.py"
)

arg_parser.add_argument("--psiblast-path", "-p",
    help="Path to psiblast executable., Necessary if psiblast is not callable by terminal",
)

arg_parser.add_argument("--workdir", "-w",
    help="""
    PSSMGen working directory including /pssm_raw and /fasta
    """,
    required=True,
) #'/projects/0/einf2380/data/pMHCII/pssm_raw/hla_drb1_0101'

arg_parser.add_argument("--blast-db", "-b",
    help="Path and name of blast db. Defaults to /projects/0/einf2380/data/blast_dbs/all_hla/all_hla",
    default='/projects/0/einf2380/data/blast_dbs/all_hla/all_hla'
)

a = arg_parser.parse_args()

if a.psiblast_path:
    psiblast = a.psiblast_path
else:
    psiblast = 'psiblast'

#create the raw_pssm, for the proof of principle work with the HLA-A*02:01 allele, only 1 sequence is required:
gen = PSSM(work_dir=a.workdir)

# set psiblast executable, database and other psiblast parameters (here shows the defaults)
gen.configure(blast_exe=psiblast,
            database=a.blast_db,
            num_threads = 3, evalue=0.0001, comp_based_stats='T',
            max_target_seqs=2000, num_iterations=3, outfmt=7,
            save_each_pssm=True, save_pssm_after_last_round=True)

# generates raw PSSM files by running BLAST with fasta files
gen.get_pssm(fasta_dir='fasta', out_dir='pssm_raw', run=True, save_all_psiblast_output=True)
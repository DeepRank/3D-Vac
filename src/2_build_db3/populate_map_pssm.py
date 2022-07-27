import glob
import os
import argparse
from mpi4py import MPI
import pandas as pd


arg_parser = argparse.ArgumentParser(description="Script to populate the pssm_mapped folder with the structure of `model` folder \n \
    and symlinks to pdbs and pssm_raw file.")
arg_parser.add_argument("--toe",
    help="type of experiment, either BA and EL. Required because it gives the subfolder in the pssm_mapped folder to work with. Default BA",
    choices = ["BA", "EL"],
    default="BA",
)
arg_parser.add_argument("--input-csv", "-i",
    help="db1 name in data/external/processed. The folder structure will be created for these entries.",
    default="BA_pMHCI.csv"
)
a = arg_parser.parse_args();

df = pd.read_csv(f"../../data/external/processed/{a.input_csv}")
folders = df["db2_folder"].tolist()
pssm_path = f"/projects/0/einf2380/data/pMHCI/pssm_mapped/{a.toe}"
raw_pssm = "/projects/0/einf2380/data/pMHCI/pssm_raw/hla_a_02_01/hla.pssm"

print(folders[0])

#stuff to parralize:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
batch = int(len(folders)/size)

if rank == size-1:
    cut_folders = len(folders) - rank*batch
else:
    cut_folders = batch

for i,folder in enumerate(cut_folders):
    ## get the best pdb structure from the molpdf scores file:
    # if i==0:
    with open(f"{folder}/molpdf_DOPE.tsv") as score_f:
        #get the pdb file (which has the best score)
        rows = [row.split("\t") for row in score_f];
        best_score = min([float(row[1]) for row in rows])
        pdb = [row[0] for row in rows if float(row[1]) == best_score][0];
        
        # create the folders:
        model_folder = "/".join(folder.split("/")[-2:])
        pssm_model_folder = f"{pssm_path}/{model_folder}";
        pssm_model_pssm_raw_folder = f"{pssm_model_folder}/pssm_raw"
        pssm_model_pdb_folder = f"{pssm_model_folder}/pdb"
        
        try:
            os.makedirs(pssm_model_folder);
        except OSError as error:
            print(f"subfolder {model_folder} already created, skipping...")
        try: 
            os.mkdir(pssm_model_pssm_raw_folder)
            os.mkdir(pssm_model_pdb_folder)
        except OSError as error:
            print(f"{pssm_model_pssm_raw_folder} or {pssm_model_pdb_folder} folder already created, skipping creation..")
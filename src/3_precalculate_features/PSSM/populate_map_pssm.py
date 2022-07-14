import glob
from pyexpat import model
import os
import argparse
from mpi4py import MPI
import math


arg_parser = argparse.ArgumentParser(description="Script to populate the pssm_mapped folder with the structure of `model` folder \n \
    and symlinks to pdbs and pssm_raw file. This script is especially adapted for the proof of principle study")
arg_parser.add_argument("--toe",
    help="type of experiment, either BA and EL. Required because it gives the subfolder in the pssm_mapped folder to work with. Default BA",
    choices = ["BA", "EL"],
    default="BA",
)
a = arg_parser.parse_args();

folders = glob.glob(f"/projects/0/einf2380/data/pMHCI/models/{a.toe}/*/*")
pssm_path = f"/projects/0/einf2380/data/pMHCI/pssm_mapped/{a.toe}"
raw_pssm = "/projects/0/einf2380/data/pMHCI/pssm_raw/hla_a_02_01/hla.pssm"

#stuff to parralize:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
step = int(len(folders)/size)
start = int(rank*step)
end = int((rank+1)*step)

if rank != size-1:
    cut_folders = folders[start:end]
else:
    cut_folders = folders[start:]

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
        model_name = pdb.split(".")[0];
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
        
        #make the symlinks
        try:  
            os.symlink(raw_pssm, f"{pssm_model_pssm_raw_folder}/{model_name}.M.pssm")
        except OSError as error:
            print(f"{pssm_model_pssm_raw_folder}/{model_name}.M.pssm symlink already exists")
        try:
            os.symlink(f"{folder}/{pdb}", f"{pssm_model_pdb_folder}/{model_name}.pdb")
        except OSError as error:
            print(f"{pssm_model_pdb_folder}/{model_name}.pdb symlink already exists")

#check that everything went fine:
# pssm_mapped_folders = glob.glob(f"{pssm_path}/*/*")

# if len(pssm_mapped_folders == len(folders)):
#     print("pssm_mapped has been successfully populated!")
# else:
#     print("the number of models does not corresponds to the number of folders created in pssm_mapped, \n \
#         something went wrong...")
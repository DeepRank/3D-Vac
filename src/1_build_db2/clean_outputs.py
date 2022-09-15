import os
from mpi4py import MPI
import argparse
import glob

arg_parser = argparse.ArgumentParser(
    description="""
    Script used to clean the models from unecessary files after `modelling_job.py` has completed.
    """
)
arg_parser.add_argument("--models-path", "-p",
    help = "glob.glob() string argument to generate a list of all models. A short tutorial on how to use glob.glob: \
    https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/\
     Default value: \
    /projects/0/einf2380/data/pMHCI/3D_models/BA/*/*",
    default = "/projects/0/einf2380/data/pMHCI/3D_models/BA/*/*"
)
a = arg_parser.parse_args()

folders = glob.glob(a.models_path)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


step = int(len(folders)/size)
start = int(rank*step+1)
end = int((rank+1)*step)

if rank != size-1:
    cut_folders = folders[start:end]
else:
    cut_folders = folders[start:]

for folder in cut_folders:
    os.system(f"rm {folder}/*.DL*")
    os.system(f'rm {folder}/*.B99990001.pdb')
    os.system(f'rm {folder}/*.V99990001')
    os.system(f'rm {folder}/*.D00000001')
    os.system(f'rm -r {folder}/__pycache__/')
    os.system(f'rm {folder}/*.lrsr')
    os.system(f'rm {folder}/*.rsr')
    os.system(f'rm {folder}/*.sch')

print(f"Cleaning on {rank} finished.")
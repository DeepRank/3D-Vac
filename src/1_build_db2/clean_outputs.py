import os
import argparse
import glob
import math

arg_parser = argparse.ArgumentParser(
    description="""
    Script used to clean the models from unecessary files after `modelling_job.py` has completed.
    """
)
arg_parser.add_argument("--models-path", "-p",
    help = "glob.glob() string argument to generate a list of all models. A short tutorial on how to use glob.glob: \
    https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/\
     Default value: \
    /projects/0/einf2380/data/pMHCI/models/BA",
    default = "/projects/0/einf2380/data/pMHCI/models/BA"
)
a = arg_parser.parse_args()

wildcard_path = os.path.join(a.models_path, '*/*')
folders = glob.glob(wildcard_path)


# determine node index so we don't do the same chunk multiple times
node_index = int(os.getenv('SLURM_NODEID'))
n_nodes = int(os.getenv('SLURM_JOB_NUM_NODES'))


step = math.ceil(len(folders)/n_nodes)
start = node_index*step
end = start + step
if end > len(folders)+1:
    end = len(folders)+1

print(f'Start: {start} \nEnd {end} \nLen folders {len(folders)}')

# step = int(len(folders)/n_nodes)
# start = int(node_index*step+1)
# end = int((node_index+1)*step)

if node_index != n_nodes-1:
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

print(f"Cleaning on node {node_index} finished.")
import glob
import pdb2sql
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sys
import os
import traceback
import argparse
from joblib import Parallel, delayed
import time

arg_parser = argparse.ArgumentParser(
    description="""
    Orients the template structure on the three first PCs of all the aligned models
    """
)
arg_parser.add_argument("--pdbs-path", "-p",
    help="""
    Folders containing the pdb models to be used for the PCA calculation
    """,
    required=True
)
arg_parser.add_argument('-c',
                    '--csv',
                    type=str,
                    required= False,
                    help='A csv to filter the pdbs that need to be aligned instead of aligning everything from db3')
arg_parser.add_argument("--n-cores", "-n",
    help="""
    Number of cores
    """,
    type=int,
	default=1
)

def get_model_coords(model):
    """get the model coordinates from the temp (intermediate) pdb
        i.e. the pdb that was rotated in the align_pdb.py
    Args:
        model (str): path of the pdb

    Returns:
        coords (list): coordinates of each atom
    """
    sql = pdb2sql.pdb2sql(model)
    coords = sql.get('resSeq, x,y,z', chainID=['P'])
    return coords

def rotate_and_save(path, pca):
    try:
        sql = pdb2sql.pdb2sql(path)
        sql_coords = sql.get('x,y,z')
        sql.update('x,y,z', pca.transform(sql_coords))
        sql.exportpdb(path)
    except:
        print(f'Something went wrong with {path}\n{traceback.format_exc()}')
    
def fast_load_dirs(sub_folder):
    pdb_files = glob.glob(os.path.join(sub_folder, '*/pdb/*.pdb'))    
    return pdb_files

n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))

a = arg_parser.parse_args()
a.pdbs_path = a.pdbs_path.replace('\\','')

sub_folders = glob.glob(os.path.join(a.pdbs_path,'*'))

    # if csv is provided
if a.csv:
    df = pd.read_csv(a.csv, header=0)
    filter_ids = df['ID'].tolist()
else:
    filter_ids = False

print('GLOB')
t0 = time.time()
pdbs_list = Parallel(n_jobs=n_cores, verbose=1)(delayed(fast_load_dirs)(pdb_sub) for pdb_sub in sub_folders)
models = [x for sublist in pdbs_list for x in sublist]
print(f'number of models found: {len(models)}')

t1 = time.time()
print( t1 - t0)

print('RETRIEVE COORDS')
all_coords = Parallel(n_jobs = a.n_cores, verbose = 1)(delayed(get_model_coords)(model) for model in models)

coords = [[x[1:] for x in y] for y in all_coords]
all_coords = []
for x in coords:
    all_coords.extend(x)
print(f'number of coordinates found: {len(all_coords)}')

t2 = time.time()
print( t2 - t1)
print('PCA')
#raise Exception(' to fix the rest of the script')
pca = PCA(n_components=3)
pca.fit(all_coords)
print(f'PCA:\n {pca.components_}')

if filter_ids:
    models = [m for m in models if os.path.basename(m).split('.')[0] in filter_ids]

print(f'total amount of models to orient with PCA: {len(models)}')

t3 = time.time()
print( t3 - t2)
print('APPLY PCA AND SAVE')
Parallel(n_jobs = a.n_cores, verbose = 1)(delayed(rotate_and_save)(path, pca) for path in models)
t4 = time.time()
print( t4 - t3)
print('DONE')

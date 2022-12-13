import glob
from itertools import _T3
import pdb2sql
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import numpy as np
import pickle
import sys
import os
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
arg_parser.add_argument("--n-cores", "-n",
    help="""
    Number of cores
    """,
    type=int,
	default=1
)

def get_model_coords(model):
    sql = pdb2sql.pdb2sql(model)
    coords = sql.get('resSeq, x,y,z', chainID=['P'])
    return coords

def rotate_and_save(path, pca):

    sql = pdb2sql.pdb2sql(path)
    sql_coords = sql.get('x,y,z')
    sql.update('x,y,z', pca.transform(sql_coords))
    sql.exportpdb(path)


a = arg_parser.parse_args()
a.pdbs_path = a.pdbs_path.replace('\\','')

t0 = time.time()
print('GLOB')
models = glob.glob(a.pdbs_path)
t1 = time.time()
print( t1 - t0)
print('RETRIEVE COORDS')
all_coords = Parallel(n_jobs = a.n_cores, verbose = 1)(delayed(get_model_coords)(model) for model in models)

coords = [[x[1:] for x in y] for y in all_coords]
all_coords = []
for x in coords:
    all_coords.extend(x)

t2 = time.time()
print( t2 - t1)
print('PCA')
#raise Exception(' to fix the rest of the script')
pca = PCA(n_components=3)
pca.fit(all_coords)

t3 = time.time()
print( t3 - t2)
print('APPLY PCA AND SAVE')
Parallel(n_jobs = a.n_cores, verbose = 1)(delayed(rotate_and_save)(path, pca) for path in models)
t4 = time.time()
print( t4 - t3)
print('DONE')

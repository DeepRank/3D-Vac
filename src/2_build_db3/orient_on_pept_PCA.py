from glob import glob
import pdb2sql
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import numpy as np
import pickle
import sys
import os
import argparse

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
arg_parser.add_argument("--template", "-t",
    help="""
    Template file to be re-oriented
    """,
    required=True
)

def get_coords(aligned_pdbs_list:list):
    """get coordinates of the peptides out of the pdbs

    Args:
        aligned_pdbs_list (list): a sublist of pdb paths (size of sublist is optimized for number of cores)

    Returns:
        all_coords (list): list of x,y,z coordinates for each pdb
    """    
    all_coords = []
    for model in [ x for x in aligned_pdbs_list if '_origin' not in x]:
        sql = pdb2sql.pdb2sql(model)
        coords = sql.get('resSeq, x,y,z', chainID=['P'])
        all_coords.append(coords)
    return all_coords


if __name__ == "__main__":
    a = arg_parser.parse_args()
    # n_cores = n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))
    n_cores = 2

    paths = glob(a.pdbs_path.replace('\\', ''))
    print(f'Found {len(paths)} pdbs in orient_on_pept_PCA')
    assert len(paths) > 0
    # get the peptides coordinates in parallel
    all_coords = Parallel(n_jobs = n_cores, verbose = 1)(delayed(get_coords)(sublist) for sublist in np.array_split(glob(a.pdbs_path), n_cores))
    # flatten nested list to get a (n,3) dim list
    flatten_coords = [item for sublist in all_coords for item in sublist]
    # just keep x,y,z values
    coords = [[x[1:] for x in y] for y in flatten_coords]

    pca = PCA(n_components=3)
    all_coords = []
    # put all coordinates in a flat list
    for x in coords:
        all_coords.extend(x)
    # fit PCA
    pca.fit(all_coords)
    # get the coordinates of the template pdb
    sql = pdb2sql.pdb2sql(a.template)
    sql_coords = sql.get('x,y,z')
    # upate the coordinates of the template pdb and write new template pdb file
    sql.update('x,y,z', pca.transform(sql_coords))
    sql.exportpdb(a.template)
import glob
import pdb2sql
from sklearn.decomposition import PCA
import pickle
import sys
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


a = arg_parser.parse_args()

all_coords = []
#TODO: this should be parallelized to optimize the resources usage
for model in [ x for x in glob.glob(a.pdbs_path) if '_origin' not in x]:
    sql = pdb2sql.pdb2sql(model)
    coords = sql.get('resSeq, x,y,z', chainID=['P'])
    # TODO: What is this if doing? Does it check for the peptide length?
    if coords[-1][0] == 15:
        all_coords.append(coords)

coords = [[x[1:] for x in y] for y in all_coords]
pca = PCA(n_components=3)
all_coords = []

for x in coords:
    all_coords.extend(x)
    
pca.fit(all_coords)

sql = pdb2sql.pdb2sql(a.template)
sql_coords = sql.get('x,y,z')

sql.update('x,y,z', pca.transform(sql_coords))

#sql.exportpdb('/projects/0/einf2380/data/pMHCII/models/alignment/alignment_template.pdb')
sql.exportpdb(a.template)

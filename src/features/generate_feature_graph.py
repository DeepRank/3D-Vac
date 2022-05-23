from deeprank_gnn.feature import bsa, pssm, amino_acid, biopython, atomic_contact, sasa
from deeprank_gnn.models.query import ProteinProteinInterfaceResidueQuery
from deeprank_gnn.preprocess import PreProcessor
import glob
import time
import os

# Get the best models from all the folders:
pdbs = [fname.split('molpdf_DOPE.tsv')[0]+open(fname).read().split('\t')[0]
		 for fname in glob.glob('pdb_models/*/*/molpdf_DOPE.tsv')]


# Retrieving the pssm's for the MHCs and the peptides:
pssmM = [glob.glob(pdb.replace('pdb_models/', 'BA/').replace(pdb.split('/')[-1],'')+'pssm/*M*.pssm')[0] for pdb in pdbs]
pssmP = [glob.glob(pdb.replace('pdb_models/', 'BA/').replace(pdb.split('/')[-1],'')+'pssm/*P*.pssm')[0] for pdb in pdbs]


# The features we are using
feature_modules = [bsa, pssm, amino_acid, biopython, atomic_contact, sasa]
preprocessor = PreProcessor(feature_modules, "output/train-data")
print(time.time()-t0)

# Add all the pdbs to the preprocessor
_ = [preprocessor.add_query(ProteinProteinInterfaceResidueQuery(pdb_path='%s' % pdb, 
		chain_id1="M", chain_id2="P", pssm_paths={"M": pssmM[it], "P": pssmP[it]})) for it, pdb in enumerate(pdbs)]

# start and save the features as hdf5
preprocessor.start()  # start builfing graphs from the queries
preprocessor.wait()  # wait for all jobs to complete
print(preprocessor.output_paths)  # print the paths of the generated files


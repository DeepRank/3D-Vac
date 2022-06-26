# This script uses deeprank-gnn-2 to generate the .hdf5 files containing graphs, that can be
# used for the training and testing of the neural networks (via NeuralNet class) contained
# in deeprank-gnn-2. It takes models' folder obtained using Pandora (.pdb files) as input
# and returns the .hdf5 files containing features and target graphs of the pMHC complexes.

from deeprankcore.feature import bsa, pssm, amino_acid, biopython, atomic_contact, sasa
from deeprankcore.models.query import ProteinProteinInterfaceResidueQuery
from deeprankcore.preprocess import preprocess
import glob
import os
​
def processCSV(csv_file):
	csv      = open(csv_file).read().split('\n')[1:-1]
	csv      = [i.split(',')for i in csv]
	csv_transposed = [[row[column] for row in csv]for column in range(len(csv[0]))]
	# return ids, alleles, peptides, scores, cluster
	return csv_transposed[8], csv_transposed[0], csv_transposed[1], csv_transposed[2], csv_transposed[10]
​
def getBestScorePdbs(pdb_models_folder):
	# Get the best models from all the folders:
	pdbs = [fname.split('molpdf_DOPE.tsv')[0]+open(fname).read().split('\t')[0]
			 for fname in glob.glob('%s/*/*/molpdf_DOPE.tsv' % pdb_models_folder)]
	return pdbs
​
def getPSSMs(best_pdb_model_list):
	# Retrieving the pssm's for the MHCs and the peptides:
	pssmM = [glob.glob(pdb.replace('pdb_models/', 'BA/').replace(pdb.split('/')[-1], '')\
			 + 'pssm/*M*.pssm')[0] for pdb in best_pdb_model_list]
	pssmP = [glob.glob(pdb.replace('pdb_models/', 'BA/').replace(pdb.split('/')[-1], '')\
			 +'pssm/*P*.pssm')[0] for pdb in best_pdb_model_list]
	return pssmM, pssmP
​
def getPilotTargets(pdb_list, pilot_data_csv='BA_pMHCI.csv'):
	# Get the targets/labels
	ids, alleles, peptides, scores, clusters = processCSV(pilot_data_csv)
	targets     = [[int(float(score) <= 500), float(score)] for i, score in enumerate(scores)]
	pdb_targets = [targets[ids.index('_'.join(pdb.split('/')[2].split('_')[:2]))] 
					for pdb in pdb_list]
	return pdb_targets, clusters
​
def makeHdf5(pilot_data_csv, pdb_models_folder, outputFolder, 
		feature_modules = [pssm, bsa, amino_acid, biopython, atomic_contact, sasa]):
	# Retrieve the pdb models with the highest score and their PSSMs
	pdbs = getBestScorePdbs(pdb_models_folder)
	pssmM, pssmP = getPSSMs(pdbs)
	# Get the targets/labels
	pdb_targets, clusters = getPilotTargets(pdbs, pilot_data_csv)
	# If outputfolder does not excist, create it
	try: os.mkdir(outputFolder)
	except:pass
	# Add all the pdbs to the preprocessor
	queries = [ProteinProteinInterfaceResidueQuery(pdb_path='%s' % pdb, 
			chain_id1="M", chain_id2="P", targets = {'binary':pdb_targets[it][0], 'BA':pdb_targets[it][1], 'cluster':clusters[it]},
			pssm_paths={"M": pssmM[it], "P": pssmP[it]}) 		
			for it, pdb in enumerate(pdbs)]
	output_paths = preprocess(feature_modules, queries, "%s/train-data" % outputFolder)
	# print the paths of the generated files
	print(output_paths)
​
if __name__ == "__main__":
	makeHdf5('BA_pMHCI.csv', 'pdb_models_folder_example', 'output_folder_example')
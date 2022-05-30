# This script uses deeprank-gnn-2 to generate the .hdf5 files containing graphs, that can be
# used for the training and testing of the neural networks (via NeuralNet class) contained
# in deeprank-gnn-2. It takes models' folder obtained using Pandora (.pdb files) as input
# and returns the .hdf5 files containing features and target graphs of the pMHC complexes.

from deeprank_gnn.feature import bsa, pssm, amino_acid, biopython, atomic_contact, sasa
from deeprank_gnn.models.query import ProteinProteinInterfaceResidueQuery
from deeprank_gnn.preprocess import PreProcessor
import glob
import time
import os

def processCSV(csv_file):
	csv      = open(csv_file).read().split('\n')[:-1]
	csv      = [i.split(',')for i in csv]
	csv_transposed = [[row[column] for row in csv]for column in range(len(csv[0]))]
	# return ids, alleles, peptides, scores
	return csv_transposed[:4]

def getBestScorePdbs(pdb_models_folder):
	# Get the best models from all the folders:
	pdbs = [fname.split('molpdf_DOPE.tsv')[0]+open(fname).read().split('\t')[0]
			 for fname in glob.glob('%s/*/*/molpdf_DOPE.tsv' % pdb_models_folder)]
	return pdbs

def getPSSMs(best_pdb_model_list):
	# Retrieving the pssm's for the MHCs and the peptides:
	pssmM = [glob.glob(pdb.replace('pdb_models/', 'BA/').replace(pdb.split('/')[-1],'')\
			 + 'pssm/*M*.pssm')[0] for pdb in best_pdb_model_list]
	pssmP = [glob.glob(pdb.replace('pdb_models/', 'BA/').replace(pdb.split('/')[-1],'')\
			 +'pssm/*P*.pssm')[0] for pdb in best_pdb_model_list]
	return pssmM, pssmP

def getPilotTargets(pdb_list, pilot_data_csv='BA_pMHCI.csv'):
	# Get the targets/labels
	ids, alleles, peptides, scores = processCSV(pilot_data_csv)
	targets     = [int(float(score) <= 500) for score in scores]
	pdb_targets = [targets[ids.index('_'.join(pdb.split('/')[2].split('_')[:2]))] 
					for pdb in pdb_list]
	return pdb_targets

def makeHdf5(pilot_data_csv, pdb_models_folder, outputFolder, feature_modules = [pssm, bsa, amino_acid, biopython, atomic_contact, sasa]):
	# init the preprocessor
	preprocessor = PreProcessor(feature_modules, "%s/train-data" % outputFolder)
	# Retrieve the pdb models with the highest score and their PSSMs
	pdbs = getBestScorePdbs(pdb_models_folder)
	pssmM, pssmP = getPSSMs(pdbs)
	# Get the targets/labels
	pdb_targets = getPilotTargets(pdbs, pilot_data_csv)
	# If outputfolder does not excist, create it
	try: os.mkdir(outputFolder)
	except:pass
	print(pdbs)
	# Add all the pdbs to the preprocessor
	_ = [preprocessor.add_query(ProteinProteinInterfaceResidueQuery(pdb_path='%s' % pdb, 
			chain_id1="M", chain_id2="P", targets = {'labels':pdb_targets[it]},
			pssm_paths={"M": pssmM[it], "P": pssmP[it]})) 		
			for it, pdb in enumerate(pdbs)]

	# start and save the features as hdf5
	preprocessor.start()  # start builfing graphs from the queries
	preprocessor.wait()  # wait for all jobs to complete
	print(preprocessor.output_paths)  # print the paths of the generated files

if __name__ == "__main__":
	makeHdf5('BA_pMHCI.csv', 'pdb_models_folder_example', 'output_folder_example')

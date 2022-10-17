from deeprankcore.feature import bsa, pssm, amino_acid, biopython, atomic_contact, sasa
from deeprankcore.preprocess import preprocess
import glob
import os
import re
import sys

def getBestScorePdbs(pdb_models_folder):
	'''
	Takes as input the folder containing the .pdb models generated by Pandora, 
	and select the best one from the .tsv file.

	The .tsv file contains the models' scores and is located in the same pdbs' folder.

	Returns a list containing such pdbs' paths.
	'''
	pdbs_list = [fname.split('molpdf_DOPE.tsv')[0]+open(fname).read().split('\t')[0] \
			 for fname in glob.glob(f'{pdb_models_folder}/*/*/molpdf_DOPE.tsv')]
	return pdbs_list

def getPssms(pdbs_list):
    '''
    Takes as input a list containing the selected pdbs' paths.
    
    Returns two lists, containing pssms' files of MHCs and peptides, respectively. 
    '''
    pssm_m = [glob.glob(pdb.replace('models/', 'pssm_mapped/').replace(pdb.split('/')[-1], '') \
			 + 'pssm/*M*.pssm')[0] for pdb in pdbs_list]
    pssm_p = [glob.glob(pdb.replace('models/', 'pssm_mapped/').replace(pdb.split('/')[-1], '') \
			 +'pssm/*P*.pssm')[0] for pdb in pdbs_list]
    return pssm_m, pssm_p

def processCsv(csv_file_path):
	'''
	Takes as input the .csv file contaning {task} data.

	Returns lists of ids, alleles, peptides, scores, cluster.
	'''
	csv = open(csv_file_path).read().split('\n')[1:-1]
	csv = [i.split(',')for i in csv]
	csv_transposed = [[row[column] for row in csv]for column in range(len(csv[0]))]
	return csv_transposed[8], csv_transposed[0], csv_transposed[1], csv_transposed[2], csv_transposed[10]

def getPilotTargets(pdbs_list, csv_file_path):
    '''
	Takes as input the pdbs list and the .csv containing {task} data.

	Returns lists of {task} targets for each pdb in pdbs_list, and the corresponding clusters. 
	'''

    pattern = r'(BA[_]\w+)[.]\w+[.]pdb'
    # Finds BA_xxxxx id in each pdb path's, and uses it to retrieve the corresponding target
    # Then pdb_targets variable will contain the targets according to pdbs_list paths' order
    pdb_ids = re.compile(pattern)
    
    ids, _, _, scores, clusters = processCsv(csv_file_path)
    # Creates binary targets, keeping also the continuous value of {task} as second element of each sublist
    targets = [[int(float(score) <= 500), float(score)] for _, score in enumerate(scores)]
    pdb_targets = [targets[ids.index(pdb_ids.findall(pdb)[0])] \
        for pdb in pdbs_list]
    return pdb_targets, clusters

if __name__ == "__main__":

	####### please modify here #######
	#run_day = '13072022'
	run_day = '17102022'
	#project_folder = '/projects/0/einf2380/'
	project_folder = '/Users/giuliacrocioni/Desktop/docs/eScience/projects/3D-vac/snellius_50/'
	data = 'pMHCI'
	task = 'BA'
	resolution = 'residue' # either 'residue' or 'atomic'
	feature_modules = [pssm, bsa, amino_acid, biopython, atomic_contact, sasa]
	interface_distance_cutoff = 15 # max distance in Å between two interacting residues/atoms of two proteins
	#process_count = 32 # remember to set the same number in --cpus-per-task in 0_generate_hdf5.sh
	process_count = 8
	##################################

	# pdb_models_folder = f'{project_folder}data/{data}/3D_models/{task}/'
	# csv_file_path = f'{project_folder}data/binding_data/{task}_{data}.csv'
	# output_folder = f'{project_folder}data/{data}/features_output_folder/GNN/{resolution}/{run_day}'
	
	pdb_models_folder = f'{project_folder}data/{data}/models/{task}/'
	csv_file_path = f'{project_folder}data/binding_data/{task}_{data}.csv'
	output_folder = f'{project_folder}data/{data}/features_output_folder/GNN/{resolution}/{run_day}'
	
	if resolution == 'atomic':
		from deeprankcore.models.query import ProteinProteinInterfaceAtomicQuery as PPIQuery
	else:
		from deeprankcore.models.query import ProteinProteinInterfaceResidueQuery as PPIQuery

	print('Script running has started ...')
	pdbs_list = getBestScorePdbs(pdb_models_folder)
	print(f'pdbs files paths loaded, {len(pdbs_list)} pdbs found.')
	pssm_m, pssm_p = getPssms(pdbs_list)
	print(f'pssms files paths loaded, {len(pssm_m)} and {len(pssm_p)} files found for M and P, respectively.\n')
	pdb_targets, clusters = getPilotTargets(pdbs_list, csv_file_path)
	print(f'Targets retrieved from {csv_file_path}, and aligned with pdbs files.\nThere are {len(pdb_targets)} targets values and {len(clusters)} cluster values. Total number of clusters is {len(set(clusters))}.\n')
	print(f'Creating output folder and adding all the listed pdbs to the preprocessor...')

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	else:
		sys.exit(f'{output_folder} already exists, please update output_folder name!')

	queries = [PPIQuery(
		pdb_path = pdb, 
		chain_id1 = "M",
		chain_id2 = "P",
		interface_distance_cutoff = interface_distance_cutoff,
		targets = {
			'binary': pdb_targets[it][0], # binary target value
			f'{task}': pdb_targets[it][1], # continuous target value
			'cluster': clusters[it]
			},
		pssm_paths = {
			"M": pssm_m[it],
			"P": pssm_p[it]
			}) \
				for it, pdb in enumerate(pdbs_list)]
	print(f'Queries created and ready to be processed.\n')
	
	# Note that preprocess() has also process_count parameter, that by default takes all available cpu cores.
	# BUT on Snellius the default will allocate 1 cpu core per task. Remember to set --cpus-per-task properly in the .sh script.
	output_paths = preprocess(feature_modules, queries, f'{output_folder}/preprocessed-data', process_count)
	print(f'Processing is done. hdf5 files generated are in {output_folder}.')
import os
os.environ['NUMEXPR_MAX_THREADS'] = '128'
import pandas as pd
import glob
import sys
from deeprankcore.query import QueryCollection
import logging

####### please modify here #######
run_day = '230201_missing_ids'
# project_folder = '/home/ccrocion/snellius_data_sample/'
#project_folder = '/Users/giuliacrocioni/Desktop/docs/eScience/projects/3D-vac/snellius_data/snellius_100_07122022/'
project_folder = '/projects/0/einf2380/'
csv_file_name = 'BA_pMHCI_human_quantitative_only_eq.csv'
models_folder_name = 'exp_nmers_all_HLA_quantitative'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
interface_distance_cutoff = 15 # max distance in Å between two interacting residues/atoms of two proteins
cpu_count = 8 # remember to set the same number in --cpus-per-task in 1_generate_hdf5.sh
verbose = False
debug_missing_ids = True
##################################

if resolution == 'atomic':
	from deeprankcore.query import ProteinProteinInterfaceAtomicQuery as PPIQuery
else:
	from deeprankcore.query import ProteinProteinInterfaceResidueQuery as PPIQuery

csv_file_path = f'{project_folder}data/external/processed/I/{csv_file_name}'
models_folder_path = f'{project_folder}data/{data}/features_input_folder/{models_folder_name}'
output_folder = f'{project_folder}data/{data}/features_output_folder/GNN/{resolution}/{run_day}'

def generate_data():
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	else:
		sys.exit(f'{output_folder} already exists, please update output_folder name!')

	# Loggers
	_log = logging.getLogger('')
	_log.setLevel(logging.INFO)

	fh = logging.FileHandler(os.path.join(output_folder, '1_generate_features.log'))
	sh = logging.StreamHandler(sys.stdout)
	fh.setLevel(logging.INFO)
	sh.setLevel(logging.INFO)
	formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
								datefmt='%a, %d %b %Y %H:%M:%S')
	fh.setFormatter(formatter_fh)

	_log.addHandler(fh)
	_log.addHandler(sh)

	_log.info('\nScript running has started ...')

	csv_file_path = f'{project_folder}data/external/processed/I/{csv_file_name}'
	csv_data = pd.read_csv(csv_file_path)
	csv_data.cluster = csv_data.cluster.fillna(-1)
	csv_data['peptide_length'] = csv_data.peptide.apply(lambda x: len(x))
	csv_data = csv_data[csv_data.peptide_length <= 15]
	csv_ids = csv_data.ID.values.tolist()

	if debug_missing_ids:
		import ast
		with open("/home/ccrocion/repositories/3D-Vac/src/3_build_db4/GNN/missing_ids.txt") as f:
			missing_ids = ast.literal_eval(f.read())
		_log.info(f'Len of missing IDs list: {len(missing_ids)}')
		pdb_files = [os.path.join(models_folder_path + '/pdb', missing_id + '.pdb') for missing_id in missing_ids]
		_log.info(f'Selected {len(pdb_files)} PDBs using missing IDs (intersection).')
	else:
		_log.info(f'Loaded CSV file containing clusters and targets data. Total number of data points is {len(csv_ids)}.')
		pdb_files_all = glob.glob(os.path.join(models_folder_path + '/pdb', '*.pdb'))
		_log.info(f'{len(pdb_files_all)} PDBs found.')
		pdb_files_csv = [os.path.join(models_folder_path + '/pdb', csv_id + '.pdb') for csv_id in csv_ids]
		pdb_files = list(set(pdb_files_all) & set(pdb_files_csv))
		pdb_files.sort()
		_log.info(f'Selected {len(pdb_files)} PDBs using CSV IDs (intersection).')

	_log.info('Aligning clusters and targets data with selected PDBs IDs ...')
	pdb_ids_csv = [pdb_file.split('/')[-1].split('.')[0] for pdb_file in pdb_files]
	csv_data_indexed = csv_data.set_index('ID')
	csv_data_indexed = csv_data_indexed.loc[pdb_ids_csv]
	assert csv_data_indexed.index.tolist() == pdb_ids_csv
	clusters = csv_data_indexed.cluster.values.tolist()
	_log.info(f'Clusters for {len(clusters)} data points loaded.')
	bas = csv_data_indexed.measurement_value.values.tolist()
	_log.info(f'Targets for {len(bas)} data points loaded.')
	pssm_m_all = glob.glob(os.path.join(models_folder_path + '/pssm', '*.M.*.pssm'))
	_log.info(f'{len(pssm_m_all)} MHC PSSMs found.')
	pssm_m_csv = [os.path.join(models_folder_path + '/pssm', csv_id + '.M.pdb.pssm') for csv_id in csv_ids]
	pssm_m = list(set(pssm_m_all) & set(pssm_m_csv))
	pssm_m.sort()
	_log.info(f'Selected {len(pssm_m)} MHC PSSMs using CSV IDs.')
	pssm_p_all = glob.glob(os.path.join(models_folder_path + '/pssm', '*.P.*.pssm'))
	_log.info(f'{len(pssm_p_all)} peptides PSSMs found.')
	pssm_p_csv = [os.path.join(models_folder_path + '/pssm', csv_id + '.P.pdb.pssm') for csv_id in csv_ids]
	pssm_p = list(set(pssm_p_all) & set(pssm_p_csv))
	pssm_p.sort()
	_log.info(f'Selected {len(pssm_p)} peptides PSSMs using CSV IDs.')

	_log.info('Verifying data consistency...')
	# verifying data consistency
	for i in range(len(pdb_files)):

		assert len(pdb_files) == len(pssm_m)
		assert len(pdb_files) == len(pssm_p)
		assert len(pdb_files) == len(clusters)
		assert len(pdb_files) == len(bas)

		try:
			assert pdb_files[i].split('/')[-1].split('.')[0] == pssm_m[i].split('/')[-1].split('.')[0]
		except AssertionError as e:
			_log.error(e)
			_log.warning(f'{pdb_files[i]} and {pssm_m[i]} ids mismatch.')

		try:
			assert pdb_files[i].split('/')[-1].split('.')[0] == pssm_p[i].split('/')[-1].split('.')[0]
		except AssertionError as e:
			_log.error(e)
			_log.warning(f'{pdb_files[i]} and {pssm_p[i]} ids mismatch.')

		try:
			assert csv_data[csv_data.ID == pdb_files[i].split('/')[-1].split('.')[0]].cluster.values[0] == clusters[i]
		except AssertionError as e:
			_log.error(e)
			_log.warning(f'{pdb_files[i]} and cluster id in the csv mismatch.')

		try:
			assert csv_data[csv_data.ID == pdb_files[i].split('/')[-1].split('.')[0]].measurement_value.values[0] == bas[i]
		except AssertionError as e:
			_log.error(e)
			_log.warning(f'{pdb_files[i]} and measurement_value id in the csv mismatch.')

	queries = QueryCollection()
	_log.info(f'Adding {len(pdb_files)} queries to the query collection ...')
	count = 0
	for i in range(len(pdb_files)):
		queries.add(
			PPIQuery(
				pdb_path = pdb_files[i], 
				chain_id1 = "M",
				chain_id2 = "P",
				distance_cutoff = interface_distance_cutoff,
				targets = {
					'binary': int(float(bas[i]) <= 500), # binary target value
					'BA': bas[i], # continuous target value
					'cluster': clusters[i]
					},
				pssm_paths = {
					"M": pssm_m[i],
					"P": pssm_p[i]
					}))
		count +=1
		if count % 10000 == 0:
			_log.info(f'{count} queries added to the collection.')

	_log.info(f'Queries ready to be processed.\n')
	output_paths = queries.process(
		f'{output_folder}/{resolution}',
		cpu_count = cpu_count,
		combine_output = False, 
		verbose = verbose)
	_log.info(f'The queries processing is done. The generated hdf5 files are in {output_folder}.')

if __name__ == "__main__":
	generate_data()

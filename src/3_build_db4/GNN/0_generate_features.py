import pandas as pd
import glob
import os
import sys
from deeprankcore.query import QueryCollection

####### please modify here #######
run_day = '23112022'
project_folder = '/home/ccrocion/snellius_data_sample/'
csv_file_name = 'BA_pMHCI_human_quantitative.csv'
models_folder_name = 'exp_nmers_all_HLA_quantitative'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
interface_distance_cutoff = 15 # max distance in Å between two interacting residues/atoms of two proteins
process_count = 8 # 32 # remember to set the same number in --cpus-per-task in 0_generate_hdf5.sh
##################################

print('Script running has started ...')

if resolution == 'atomic':
	from deeprankcore.query import ProteinProteinInterfaceAtomicQuery as PPIQuery
else:
	from deeprankcore.query import ProteinProteinInterfaceResidueQuery as PPIQuery

csv_file_path = f'{project_folder}data/external/processed/I/{csv_file_name}'
models_folder_path = f'{project_folder}data/{data}/features_input_folder/{models_folder_name}'
output_folder = f'{project_folder}data/pMHCI/features_output_folder/GNN/{resolution}/{run_day}'
pdb_files = glob.glob(os.path.join(models_folder_path + '/pdb', '*.pdb'))
csv_data = pd.read_csv(csv_file_path)
csv_data.cluster = csv_data.cluster.fillna(-1)
print(f'pdbs files paths loaded, {len(pdb_files)} pdbs found.')

pssms_m = []
pssms_p = []
clusters = []
bas = []

for pdb_file in pdb_files:

    pdb_id = pdb_file.split('/')[-1].split('.')[0]
    pssm_m = glob.glob(os.path.join(models_folder_path + '/pssm', pdb_id + '.M.*.pssm'))
    pssm_p = glob.glob(os.path.join(models_folder_path + '/pssm', pdb_id + '.P.*.pssm'))
    assert len(pssm_m) == 1
    assert len(pssm_p) == 1
    pssms_m.append(pssm_m[0])
    pssms_p.append(pssm_p[0])

    csv_id = pdb_id.replace('-', '_')
    assert csv_data[csv_data.ID == csv_id].shape[0] == 1
    cluster = csv_data[csv_data.ID == csv_id].cluster.values[0]
    ba = csv_data[csv_data.ID == csv_id].measurement_value.values[0]
    clusters.append(cluster)
    bas.append(ba)

bas_bin = [int(float(ba) <= 500) for ba in bas]

assert len(pssms_m) == len(pdb_files)
assert len(pssms_p) == len(pdb_files)
assert len(clusters) == len(pdb_files)
assert len(bas) == len(pdb_files)
assert len(bas_bin) == len(pdb_files)

if not os.path.exists(output_folder):
	os.makedirs(output_folder)
else:
	sys.exit(f'{output_folder} already exists, please update output_folder name!')

queries = QueryCollection()

for idx, pdb_file in enumerate(pdb_files):
	queries.add(
		PPIQuery(
			pdb_path = pdb_file, 
			chain_id1 = "M",
			chain_id2 = "P",
			distance_cutoff = interface_distance_cutoff,
			targets = {
				'binary': bas_bin[idx], # binary target value
				'BA': bas[idx], # continuous target value
				'cluster': clusters[idx]
				},
			pssm_paths = {
				"M": pssms_m[idx],
				"P": pssms_p[idx]
				}))

print(f'Queries created and ready to be processed.\n')

# Note that preprocess() has also process_count parameter, that by default takes all available cpu cores.
# BUT on Snellius the default will allocate 1 cpu core per task. Remember to set --cpus-per-task properly in the .sh script.
output_paths = queries.process(f'{output_folder}/processed-data', process_count)
print(f'Processing is done. hdf5 files generated are in {output_folder}.')

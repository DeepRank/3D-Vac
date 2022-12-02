import pandas as pd
import glob
import os
import sys
from deeprankcore.query import QueryCollection
import logging

####### please modify here #######
run_day = '02122022'
project_folder = '/home/ccrocion/snellius_data_sample/'
csv_file_name = 'BA_pMHCI_human_quantitative.csv'
models_folder_name = 'exp_nmers_all_HLA_quantitative'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
interface_distance_cutoff = 15 # max distance in Å between two interacting residues/atoms of two proteins
cpu_count = 8 # 32 # remember to set the same number in --cpus-per-task in 0_generate_hdf5.sh
##################################

# Loggers
_log = logging.getLogger('')
_log.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
_log.addHandler(sh)

_log.info('Script running has started ...')

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

if not os.path.exists(output_folder):
	os.makedirs(output_folder)
else:
	sys.exit(f'{output_folder} already exists, please update output_folder name!')

_log.info(f'pdbs files paths loaded, {len(pdb_files)} pdbs found.')

queries = QueryCollection()

for pdb_file in pdb_files:

    pdb_id = pdb_file.split('/')[-1].split('.')[0]
    pssm_m = glob.glob(os.path.join(models_folder_path + '/pssm', pdb_id + '.M.*.pssm'))
    pssm_p = glob.glob(os.path.join(models_folder_path + '/pssm', pdb_id + '.P.*.pssm'))
    assert len(pssm_m) == 1
    assert len(pssm_p) == 1
    csv_id = pdb_id.replace('-', '_')
    assert csv_data[csv_data.ID == csv_id].shape[0] == 1
    cluster = csv_data[csv_data.ID == csv_id].cluster.values[0]
    ba = csv_data[csv_data.ID == csv_id].measurement_value.values[0]

    queries.add(
        PPIQuery(
            pdb_path = pdb_file, 
            chain_id1 = "M",
            chain_id2 = "P",
            distance_cutoff = interface_distance_cutoff,
            targets = {
                'binary': int(float(ba) <= 500), # binary target value
                'BA': ba, # continuous target value
                'cluster': cluster
                },
            pssm_paths = {
                "M": pssm_m[0],
                "P": pssm_p[0]
                }))

_log.info(f'Queries created and ready to be processed.\n')

output_paths = queries.process(f'{output_folder}/processed-data')

_log.info(f'Processing is done. hdf5 files generated are in {output_folder}.')

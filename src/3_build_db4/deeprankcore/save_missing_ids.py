import h5py
import glob
import os
import pandas as pd
import re

##########
run_day = '230515'
# project_folder = '/home/ccrocion/snellius_data_sample/'
project_folder = '/projects/0/einf2380/'
csv_file_name = 'BA_pMHCI_human_quantitative_only_eq.csv'
models_folder_name = 'HLA_quantitative'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
###########

output_folder = f'{project_folder}data/{data}/features_output_folder/deeprankcore/{resolution}/{run_day}'
csv_file_path = f'{project_folder}data/external/processed/I/{csv_file_name}'
models_folder_path = f'{project_folder}data/{data}/features_input_folder/{models_folder_name}'
hdf5_files = glob.glob(os.path.join(output_folder, '*.hdf5'))
csv_data = pd.read_csv(csv_file_path)
csv_ids = csv_data.ID.values.tolist()
pdb_files_all = glob.glob(os.path.join(models_folder_path + '/pdb', '*.pdb'))
pdb_files_csv = [os.path.join(models_folder_path + '/pdb', csv_id + '.pdb') for csv_id in csv_ids]
pdb_files = list(set(pdb_files_all) & set(pdb_files_csv))
pdb_files.sort()
print(f'Expected number of data points: {len(pdb_files)}.')

mol_keys = []
print('Reading in generated data points...')
count = 0
for fname in hdf5_files:
    with h5py.File(fname, 'r') as f:
        mol_keys += list(f.keys())
    count +=1
    if count % 10 == 0:
        print(f'{count} hdf5 files read. Data points cumulated: {len(mol_keys)}')

print(f'Data points generated: {len(mol_keys)}.')
print(f'Number of missing data points: {len(pdb_files) - len(mol_keys)}.')
hdf5_ids = [re.search('.+:M-P:(BA-.+)', mol).group(1) for mol in mol_keys]
pdb_ids_csv = [pdb_file.split('/')[-1].split('.')[0] for pdb_file in pdb_files]
missing_ids = list(sorted(set(pdb_ids_csv) - set(hdf5_ids)))
print(f'Length of missing data points list: {len(missing_ids)} (should match the number above.')

with open("missing_ids.txt", "w") as output:
    output.write(str(missing_ids))

print(f'IDs saved in missing_ids.txt.')

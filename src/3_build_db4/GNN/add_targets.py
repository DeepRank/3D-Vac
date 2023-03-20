# Script to add targets (e.g. clusters, alleles) to HDF5 files from a given csv file
# (in which the ids must match the ones contained in the already generated HDF5 files)
import pandas as pd
import glob
import h5py
import os
import re
from collections import Counter
import logging
import sys

############# Modify
run_day_data = '230202' # 100k data points (proj folder)
# run_day_data = '230130' # 692 data points (local folder)
protein_class = 'I'
target_data = 'BA'
resolution_data = 'residue' # either 'residue' or 'atomic'
project_folder = '/projects/0/einf2380'
# project_folder = '/home/ccrocion/snellius_data_sample' # local resized df path
hdf5_group = 'target_values/cluster'
csv_file_path = '/projects/0/einf2380/data/external/processed/I/clusters/BA_pMHCI_human_quantitative_all_hla_gibbs_clusters.csv'
cluster_col = 'cluster_set_10'
#############
folder_data = f'{project_folder}/data/pMHC{protein_class}/features_output_folder/GNN/{resolution_data}/{run_day_data}'
input_data_path = glob.glob(os.path.join(folder_data, '*.hdf5'))
output_folder = f'{project_folder}/data/pMHC{protein_class}/features_output_folder/GNN/{resolution_data}/{run_day_data}'
csv_data = pd.read_csv(csv_file_path)
csv_data['allele_type'] = csv_data.allele.str.extract(r'HLA-(\w)\*.+')
log_file_name = hdf5_group.split('/')[1]


def add_targets():
    # Loggers
    _log = logging.getLogger('')
    _log.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(output_folder, f'add_target:{log_file_name}.log'))
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
                                datefmt='%a, %d %b %Y %H:%M:%S')
    fh.setFormatter(formatter_fh)

    _log.addHandler(fh)
    _log.addHandler(sh)

    _log.info('\nScript running has started ...\n')

    for fname in input_data_path:
        try:
            with h5py.File(fname, 'r+') as hdf5:
                for mol in hdf5.keys():
                    p = re.compile('residue-ppi-(\w+-\d+):M-P')
                    csv_id = p.findall(mol)[0]
                    cluster_value = csv_data.loc[csv_data['ID'] == csv_id][cluster_col].values[0]
                    hdf5[mol][hdf5_group][()] = cluster_value

        except Exception as e:
            _log.error(e)
            _log.info(f'Error in opening {fname}, please check the file.')

    # verify
    csv_cluster = []
    hdf5_cluster = []
    count = 0
    for fname in input_data_path:
        with h5py.File(fname, 'r') as hdf5:
            for mol in hdf5.keys():
                p = re.compile('residue-ppi-(\w+-\d+):M-P')
                csv_id = p.findall(mol)[0]
                cluster_value = csv_data.loc[csv_data['ID'] == csv_id][cluster_col].values[0]
                if not cluster_value == hdf5[mol][hdf5_group][()]:
                    _log.warning(f'\nSomething went wrong for data point with id {mol} in {fname}.')
                    _log.info(f'HDF5 value: {hdf5[mol][hdf5_group][()]}')
                    _log.info(f'csv cluster value {cluster_value}')
                csv_cluster.append(cluster_value)
                hdf5_cluster.append(hdf5[mol][hdf5_group][()])
                count +=1
                if count % 10000 == 0:
                    _log.info(f'{count} data points modified.')
    _log.info(f'\n{cluster_col} values in the csv corresponding to data in the HDF5 files: {Counter(csv_cluster)}\n')
    _log.info(f'Number of data points in the HDF5 files: {len(hdf5_cluster)}\n')
    _log.info(f'{cluster_col} values in the HDF5 files: {Counter(hdf5_cluster)}')

if __name__ == "__main__":
	add_targets()

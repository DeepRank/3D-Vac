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

# ONGOING: add cl_peptide2 for 100k (2820744), using gpu bc fat partition was giving errors
# TODO: add cl_allele for 100k
# TODO: add allele_type for 100k

############ Modify
run_day_data = '230530' # 100k and 692 data points
project_folder = '/projects/0/einf2380'
# project_folder = '/home/ccrocion/snellius_data_sample' # local resized df path
# Group name in the hdf5 files
hdf5_target_group = 'target_values'
# clustering target Dataset name to be added to the hdf5 files
hdf5_target_cl = 'cl_peptide2' #'cl_peptide' # 'cl_peptide2' # 'cl_allele' # 'allele_type'
# csv file containing the clustering
csv_file_cl = 'BA_pMHCI_human_quantitative_clustered_peptides_marieke_fixed.csv' 
# 'BA_pMHCI_human_quantitative_all_hla_gibbs_clusters.csv' for cl_peptide
# 'BA_pMHCI_human_quantitative_clustered_peptides_marieke_fixed.csv' for cl_peptide2
# 'BA_pMHCI_human_quantitative_only_eq_alleleclusters_pseudoseq.csv' for cl_allele
# 'BA_pMHCI_human_quantitative_only_eq.csv' for allele_type
# clustering col name in the csv file
csv_target_col = 'Marieke_cluster' # 'cluster_set_10' # 'Marieke_cluster' # 'allele_clustering' # 'allele_type'
protein_class = 'I'
target_data = 'BA'
resolution_data = 'residue' # either 'residue' or 'atomic'
#############

if csv_target_col == 'allele_type':
    csv_file_cl_path = f'{project_folder}/data/external/processed/I/{csv_file_cl}'
else:
    csv_file_cl_path = f'{project_folder}/data/external/processed/I/clusters/{csv_file_cl}'
hdf5_target_path = hdf5_target_group + '/' + hdf5_target_cl
folder_data = f'{project_folder}/data/pMHC{protein_class}/features_output_folder/deeprankcore/{resolution_data}/{run_day_data}'
input_data_path = glob.glob(os.path.join(folder_data, '*.hdf5'))
output_folder = f'{project_folder}/data/pMHC{protein_class}/features_output_folder/deeprankcore/{resolution_data}/{run_day_data}'
csv_data = pd.read_csv(csv_file_cl_path)
if hdf5_target_cl == 'allele_type':
    csv_data[hdf5_target_cl] = csv_data.allele.str.extract(r'HLA-(\w)\*.+')
log_file_name = hdf5_target_path.split('/')[1]

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

    count = 0
    for fname in input_data_path:
        try:
            with h5py.File(fname, 'r+') as hdf5:
                for mol in hdf5.keys():
                    p = re.compile('residue-ppi:M-P:(\w+-\d+)')
                    csv_id = p.findall(mol)[0]
                    target_value = csv_data.loc[csv_data['ID'] == csv_id][csv_target_col].values[0]
                    if hdf5_target_path in hdf5[mol]:
                        hdf5[mol][hdf5_target_path][()] = target_value
                    else:
                        hdf5[mol][hdf5_target_group].create_dataset(name=hdf5_target_cl, data=target_value)
                    count +=1
                    if count % 10000 == 0:
                        _log.info(f'{count} data points modified.')

        except Exception as e:
            _log.error(e)
            _log.info(f'Error in opening {fname}, please check the file.')

    # verify
    csv_target = []
    hdf5_target = []
    count = 0
    _log.info("\nStarting verification...")
    for fname in input_data_path:
        with h5py.File(fname, 'r') as hdf5:
            for mol in hdf5.keys():
                p = re.compile('residue-ppi:M-P:(\w+-\d+)')
                csv_id = p.findall(mol)[0]
                target_csv_value = csv_data.loc[csv_data['ID'] == csv_id][csv_target_col].values[0]
                if csv_target_col == 'allele_type':
                    target_hdf5_value = hdf5[mol][hdf5_target_path].asstr()[()]
                else:
                    target_hdf5_value = hdf5[mol][hdf5_target_path][()]
                if not target_csv_value == target_hdf5_value:
                    _log.warning(f'\nSomething went wrong for data point with id {mol} in {fname}.')
                    _log.info(f'HDF5 value: {hdf5[mol][hdf5_target_path][()]}')
                    _log.info(f'CSV cluster value {target_value}')
                csv_target.append(target_csv_value)
                hdf5_target.append(target_hdf5_value)
                count +=1
                if count % 10000 == 0:
                    _log.info(f'{count} data points verified.')
    _log.info(f'Number of data points in the CSV file used in the HDF5 files: {len(csv_target)}')
    _log.info(f'{csv_target_col} values in the CSV corresponding to data in the HDF5 files:\n{Counter(csv_target)}\n')
    _log.info(f'Number of data points in the HDF5 files: {len(hdf5_target)}')
    _log.info(f'{hdf5_target_cl} values in the HDF5 files:\n{Counter(hdf5_target)}')

if __name__ == "__main__":
	add_targets()

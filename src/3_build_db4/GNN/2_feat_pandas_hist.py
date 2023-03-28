import glob
import os
import sys
import time
import logging
from deeprankcore.dataset import GraphDataset

run_day = '230328'
project_folder = '/projects/0/einf2380/'
# project_folder = '/home/ccrocion/snellius_data_sample/'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
target_dataset = 'binary'
output_folder = f'{project_folder}data/{data}/features_output_folder/GNN/{resolution}/{run_day}'

# Loggers
_log = logging.getLogger('')
_log.setLevel(logging.INFO)

fh = logging.FileHandler(os.path.join(output_folder, '2_feat_pandas_hist.log'))
sh = logging.StreamHandler(sys.stdout)
fh.setLevel(logging.INFO)
sh.setLevel(logging.INFO)
formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
                               datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter_fh)

_log.addHandler(fh)
_log.addHandler(sh)

hdf5_files = glob.glob(os.path.join(output_folder, '*.hdf5'))
_log.info(f"{len(hdf5_files)} hdf5 files found.")
hdf5_pandas = os.path.join(output_folder, f'{resolution}_pandas.feather')
images_path = os.path.join(output_folder, 'images')
if not os.path.exists(images_path):
    os.makedirs(images_path)

dataset = GraphDataset(
    hdf5_path = hdf5_files,
    target = target_dataset
)

start = time.perf_counter()
df = dataset.hdf5_to_pandas()
finish = time.perf_counter()
_log.info(f"Loading to pandas {len(hdf5_files)} hdf5 files took {round(finish-start, 2)} seconds.")

start = time.perf_counter()
df.to_feather(hdf5_pandas)
finish = time.perf_counter()
_log.info(f"df saved in {hdf5_pandas} in {round(finish-start, 2)} seconds.")

start = time.perf_counter()

count = 1
for idx in range(1, len(df.columns[1:]), 7):
    features = list(df.columns[idx:idx+7])
    dataset.save_hist(features, os.path.join(images_path, f'feat_group_{count}_rice.png'), 'rice')
    _log.info(f'Saved group {count}.')
    count += 1

finish = time.perf_counter()
_log.info(f"Distributions images saved in {round(finish-start, 2)} seconds.")

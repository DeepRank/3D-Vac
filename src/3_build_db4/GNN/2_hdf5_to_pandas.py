import glob
import os
import sys
import time
from deeprankcore.tools import hist
import logging

run_day = '230123'
project_folder = '/projects/0/einf2380/'
csv_file_name = 'BA_pMHCI_human_quantitative_only_eq.csv'
models_folder_name = 'exp_nmers_all_HLA_quantitative'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
output_folder = f'{project_folder}data/{data}/features_output_folder/GNN/{resolution}/{run_day}'

# Loggers
_log = logging.getLogger('')
_log.setLevel(logging.INFO)

fh = logging.FileHandler(os.path.join(output_folder, '2_hdf5_to_pandas.log'))
sh = logging.StreamHandler(sys.stdout)
fh.setLevel(logging.INFO)
sh.setLevel(logging.INFO)
formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
                               datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter_fh)

_log.addHandler(fh)
_log.addHandler(sh)

csv_file_path = f'{project_folder}data/external/processed/I/{csv_file_name}'
hdf5_files = glob.glob(os.path.join(output_folder, '*.hdf5'))
hdf5_pandas = os.path.join(output_folder, f'{resolution}_pandas.feather')
_log.info(f"{len(hdf5_files)} hdf5 files found.")

start = time.perf_counter()
df = hist.hdf5_to_pandas(hdf5_files, target_features='binary')
finish = time.perf_counter()
_log.info(f"Loading to pandas {len(hdf5_files)} hdf5 files took {round(finish-start, 2)} seconds.")

start = time.perf_counter()
df.to_feather(hdf5_pandas)
finish = time.perf_counter()
_log.info(f"df saved in {hdf5_pandas} in {round(finish-start, 2)} seconds.")

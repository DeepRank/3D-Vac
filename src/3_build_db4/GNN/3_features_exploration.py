import pandas as pd
import os
from deeprankcore.tools import hist
import logging
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

run_day = '230123'
project_folder = '/projects/0/einf2380/'
models_folder_name = 'exp_nmers_all_HLA_quantitative'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
output_folder = f'{project_folder}data/{data}/features_output_folder/GNN/{resolution}/{run_day}'
images_path = os.path.join(output_folder, 'images')
hdf5_pandas = os.path.join(output_folder, f'{resolution}_pandas.feather')

if not os.path.exists(images_path):
    os.makedirs(images_path)

# Loggers
_log = logging.getLogger('')
_log.setLevel(logging.INFO)

fh = logging.FileHandler(os.path.join(output_folder, '3_features_exploration.log'))
sh = logging.StreamHandler(sys.stdout)
fh.setLevel(logging.INFO)
sh.setLevel(logging.INFO)

formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
                               datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter_fh)

_log.addHandler(fh)
_log.addHandler(sh)

################################

_log.info('Script started.')

start = time.perf_counter()
df = pd.read_feather(hdf5_pandas, use_threads=True)
finish = time.perf_counter()
_log.info(f"Dataframe read in {round(finish-start, 2)} seconds.")

_log.info(f'Shape: {df.shape}')
_log.info(f'Columns: {list(df.columns)}')

start = time.perf_counter()

count = 1
for idx in range(1, len(df.columns[1:]), 7):
    features = list(df.columns[idx:idx+7])
    fig = hist.save_hist(df, features, os.path.join(images_path, f'feat_group_{count}_rice.png'), 'rice')
    _log.info(f'Saved group {count}.')
    count += 1

finish = time.perf_counter()
_log.info(f"Distributions images saved in {round(finish-start, 2)} seconds.")

# one_hot_col = [
#     'binary', 'same_chain', 'covalent', 'res_type_0', 'res_type_1', 'res_type_2',
#     'res_type_3', 'res_type_4', 'res_type_5', 'res_type_6', 'res_type_7', 'res_type_8',
#     'res_type_9', 'res_type_10', 'res_type_11', 'res_type_12', 'res_type_13', 'res_type_14',
#     'res_type_15', 'res_type_16', 'res_type_17', 'res_type_18', 'res_type_19', 'polarity_0',
#     'polarity_1', 'polarity_2', 'polarity_3']
# discrete_col = [
#     'res_size', 'pssm_0', 'pssm_1', 'pssm_2', 'pssm_3', 'pssm_4', 'pssm_5', 'pssm_6', 'pssm_7',
#     'pssm_8', 'pssm_9', 'pssm_10', 'pssm_11', 'pssm_12', 'pssm_13', 'pssm_14', 'pssm_15', 'pssm_16',
#     'pssm_17', 'pssm_18', 'pssm_19', 'hse_0', 'hse_2', 'hb_donors', 'hb_acceptors', 'charge']
# continuous_col = []


# for col in df.columns:
#     if (col not in one_hot_col and col not in discrete_col and col != 'id'):
#         continuous_col.append(col)

# def std_scaler(x, u, s):
#     return (x-u)/s

# for col in continuous_col:
#     fig = transform.plot_distr(df, col)
#     fig.write_image(f'images/{col}.png')

# for col in continuous_col:
#     u = df[col].apply(lambda x: x.mean()).mean()
#     s = df[col].apply(lambda x: x.std()).mean()
#     df[col] = df[col].apply(lambda x: std_scaler(x, u, s))

# for col in continuous_col:
#     fig = transform.plot_distr(df, col)
#     fig.write_image(f'images/{col}_std.png')

import os
import time
import pandas as pd
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import sys


####### please modify here #######
run_day = '230530'
# project_folder = '/projects/0/einf2380/'
project_folder = '/home/ccrocion/snellius_data_sample/'
data = 'pMHCI'
resolution = 'residue' # either 'residue' or 'atomic'
target_dataset = 'binary'
features = 'electrostatic'
output_folder = f'{project_folder}data/{data}/features_output_folder/deeprankcore/{resolution}/{run_day}'
##################################

# Loggers
_log = logging.getLogger('')
_log.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
                               datefmt='%a, %d %b %Y %H:%M:%S')
_log.addHandler(sh)
###############

_log.info("Script started")

hdf5_pandas = os.path.join(output_folder, f'{resolution}_pandas.feather')
images_path = os.path.join(output_folder, 'images')
if not os.path.exists(images_path):
    os.makedirs(images_path)

_log.info("Reading the pd dataframe")
df = pd.read_feather(hdf5_pandas)
_log.info("df reading done")

def save_hist( # pylint: disable=too-many-arguments, too-many-branches, useless-suppression
        df,
        features: Union[str,List[str]],
        fname: str = 'features_hist.png',
        bins: Union[int,List[float],str] = 10,
        figsize: Tuple = (15, 15),
        log: bool = False
):
    
    if not isinstance(features, list):
        features = [features]

    features_df = [col for feat in features for col in df.columns.values.tolist() if feat in col]
    
    means = [
        round(np.concatenate(df[feat].values).mean(), 1) if isinstance(df[feat].values[0], np.ndarray) \
        else round(df[feat].values.mean(), 1) \
        for feat in features_df]
    devs = [
        round(np.concatenate(df[feat].values).std(), 1) if isinstance(df[feat].values[0], np.ndarray) \
        else round(df[feat].values.std(), 1) \
        for feat in features_df]

    if len(features_df) > 1:

        fig, axs = plt.subplots(len(features_df), figsize=figsize)

        for row, feat in enumerate(features_df):       
            if isinstance(df[feat].values[0], np.ndarray):
                if(log):
                    log_data = np.log(np.concatenate(df[feat].values))
                    log_data[log_data == -np.inf] = 0
                    axs[row].hist(log_data, bins=bins)
                else:
                    axs[row].hist(np.concatenate(df[feat].values), bins=bins)
            else:
                if(log):
                    log_data = np.log(df[feat].values)
                    log_data[log_data == -np.inf] = 0 
                    axs[row].hist(log_data, bins=bins)
                else:
                    axs[row].hist(df[feat].values, bins=bins)
            axs[row].set(xlabel=f'{feat} (mean {means[row]}, std {devs[row]})', ylabel='Count')
        fig.tight_layout()

    elif len(features_df) == 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if isinstance(df[features_df[0]].values[0], np.ndarray):
            if(log):
                log_data = np.log(np.concatenate(df[features_df[0]].values))
                log_data[log_data == -np.inf] = 0
                ax.hist(log_data, bins=bins)
            else:
                ax.hist(np.concatenate(df[features_df[0]].values), bins=bins)
        else:
            if(log):
                log_data = np.log(df[features_df[0]].values)
                log_data[log_data == -np.inf] = 0
                ax.hist(log_data, bins=bins)
            else:
                ax.hist(df[features_df[0]].values, bins=bins)
        ax.set(xlabel=f'{features_df[0]} (mean {means[0]}, std {devs[0]})', ylabel='Count')

    else:
        raise ValueError("Please provide valid features names. They must be present in the current :class:`DeeprankDataset` children instance.")
    
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

save_hist(df, features, os.path.join(images_path, f'electrostatic_rice.png'), 'rice')
_log.info("histogram saved")
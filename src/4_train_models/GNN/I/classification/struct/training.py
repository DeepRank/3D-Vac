import h5py
import glob
import os
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import logging
from deeprankcore.Trainer import Trainer
from deeprankcore.ginet import GINet
from deeprankcore.naive_gnn import NaiveNetwork
from deeprankcore.DataSet import HDF5DataSet, save_hdf5_keys

# set random seed!!!

#################### To fill
# Input data
protein_class = 'I'
target_data = 'BA'
resolution_data = 'residue' # either 'residue' or 'atomic'
run_day_data = '13072022'
# Target/s
target_group = 'score/'
target_dataset = 'binary'
task = 'classif'
# Features
node_features = ['bsa', 'depth', 'hb_acceptors', 'hb_donors', 'hse', 'ic', 'polarity', 'pos', 'pssm', 'sasa', 'size', 'type']
edge_features = ['coulomb', 'covalent', 'dist', 'vanderwaals']
# Clusters
cluster_dataset = 'cluster'
train_clusters = [0, 1, 2, 3, 4, 7, 9]
val_clusters = [5, 8]
test_clusters = [6]
# Trainer
net = NaiveNetwork
task = 'classif'
batch_size = 16
optimizer = torch.optim.Adam
lr = 1e-3
weight_decay = 0
epochs = 10
save_model = 'best'
# Paths
#project_folder = '/Users/giuliacrocioni/Desktop/docs/eScience/projects/3D-vac/snellius_50/' # local resized df path
project_folder = '/projects/0/einf2380/'
folder_data = f'{project_folder}data/pMHC{protein_class}/features_output_folder/GNN/{resolution_data}/{run_day_data}'
input_data_path = folder_data + '/' + resolution_data + '.hdf5'
####################

#################### Folders and logger
# Outputs folder
exp_list = [f for f in glob.glob("experiments/exp*")]
if len(exp_list) > 0:
    nums = [int(w.split('/exp')[1]) for w in exp_list]
    exp_id = 'exp' + str(max(nums) + 1)
    exp_path = os.path.join('./experiments', exp_id)
    os.makedirs(exp_path)
else:
    exp_id = 'exp0'
    exp_path = os.path.join('./experiments', exp_id)
    os.makedirs(exp_path)

data_path = os.path.join(exp_path, 'data')
metrics_path = os.path.join(exp_path, 'metrics')
img_path = os.path.join(exp_path, 'images')
os.makedirs(data_path)
os.makedirs(metrics_path)
os.makedirs(img_path)
# Loggers
_log = logging.getLogger('')
_log.setLevel(logging.INFO)

fh = logging.FileHandler(os.path.join(exp_path, 'training.log'))
sh = logging.StreamHandler(sys.stdout)
fh.setLevel(logging.INFO)
sh.setLevel(logging.INFO)
formatter_fh = logging.Formatter('[%(asctime)s] - %(name)s - %(message)s',
                               datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter_fh)

_log.addHandler(fh)
_log.addHandler(sh)
####################

_log.info(f'Created folder {exp_path}\n')

_log.info("training.py has started!\n")

#################### Data summary
summary = {}
summary['entry'] = []
summary['cluster'] = []
summary['target'] = []
summary['phase'] = []

# '/Users/giuliacrocioni/remote_snellius/data/pMHCI/features_output_folder/GNN/residue/13072022/residue.hdf5'
with h5py.File(input_data_path, 'r') as hdf5:

    for mol in hdf5.keys():
        cluster_value = float(hdf5[mol][target_group][cluster_dataset][()])
        target_value = float(hdf5[mol][target_group][target_dataset][()])

        summary['entry'].append(mol)
        summary['cluster'].append(cluster_value)
        summary['target'].append(target_value)

        if cluster_value in train_clusters:
            summary['phase'].append('train')
        elif cluster_value in val_clusters:
            summary['phase'].append('valid')
        elif cluster_value in test_clusters:
            summary['phase'].append('test')

df_summ = pd.DataFrame(data=summary)

df_summ.to_hdf(
    os.path.join(metrics_path, 'summary_data.hdf5'),
    key='summary',
    mode='w')

df_train = df_summ[df_summ.phase == 'train']
df_valid = df_summ[df_summ.phase == 'valid']
df_test = df_summ[df_summ.phase == 'test']

_log.info(f'Data statistics:\n')
_log.info(f'Total samples: {len(df_summ)}')
_log.info(f'Training set: {len(df_train)} samples, {round(100*len(df_train)/len(df_summ))}%')
_log.info(f'\t- Class 0: {len(df_train[df_train.target == 0])} samples, {round(100*len(df_train[df_train.target == 0])/len(df_train))}%')
_log.info(f'\t- Class 1: {len(df_train[df_train.target == 1])} samples, {round(100*len(df_train[df_train.target == 1])/len(df_train))}%')
_log.info(f'Validation set: {len(df_valid)} samples, {round(100*len(df_valid)/len(df_summ))}%')
_log.info(f'\t- Class 0: {len(df_valid[df_valid.target == 0])} samples, {round(100*len(df_valid[df_valid.target == 0])/len(df_valid))}%')
_log.info(f'\t- Class 1: {len(df_valid[df_valid.target == 1])} samples, {round(100*len(df_valid[df_valid.target == 1])/len(df_valid))}%')
_log.info(f'Testing set: {len(df_test)} samples, {round(100*len(df_test)/len(df_summ))}%')
_log.info(f'\t- Class 0: {len(df_test[df_test.target == 0])} samples, {round(100*len(df_test[df_test.target == 0])/len(df_test))}%')
_log.info(f'\t- Class 1: {len(df_test[df_test.target == 1])} samples, {round(100*len(df_test[df_test.target == 1])/len(df_test))}%')

for cl in sorted(df_summ.cluster.unique(), reverse=True):
    if len(df_summ[df_summ.cluster == cl]):
        _log.info(f'\t\tCluster {int(cl)}: {len(df_summ[df_summ.cluster == cl])} samples, {round(100*len(df_summ[df_summ.cluster == cl])/len(df_summ))}%')
        _log.info(f'\t\t\t- Class 0: {len(df_summ[(df_summ.cluster == cl) & (df_summ.target == 0)])} samples, {round(100*len(df_summ[(df_summ.cluster == cl) & (df_summ.target == 0)])/len(df_summ[df_summ.cluster == cl]))}%')
        _log.info(f'\t\t\t- Class 1: {len(df_summ[(df_summ.cluster == cl) & (df_summ.target == 1)])} samples, {round(100*len(df_summ[(df_summ.cluster == cl) & (df_summ.target == 1)])/len(df_summ[df_summ.cluster == cl]))}%')
    else:
        _log.info(f'Cluster {int(cl)} not present!')

save_hdf5_keys(input_data_path, df_summ[df_summ.phase == 'train'].entry.to_list(), os.path.join(data_path, 'train.hdf5'), hardcopy = True)
save_hdf5_keys(input_data_path, df_summ[df_summ.phase == 'valid'].entry.to_list(), os.path.join(data_path, 'valid.hdf5'), hardcopy = True)
save_hdf5_keys(input_data_path, df_summ[df_summ.phase == 'test'].entry.to_list(), os.path.join(data_path, 'test.hdf5'), hardcopy = True)
####################

#################### HDF5DataSet

_log.info(f'HDF5DataSet loading...\n')
# to change: pass in only list of keys
dataset_train = HDF5DataSet(
    hdf5_path = [
        os.path.join(data_path, 'train.hdf5')],
    target = target_dataset,
    task = task,
    node_feature = node_features,
    edge_feature = edge_features
)
dataset_val = HDF5DataSet(
    hdf5_path = [
        os.path.join(data_path, 'valid.hdf5')],
    target = target_dataset,
    task = task,
    node_feature = node_features,
    edge_feature = edge_features
)
dataset_test = HDF5DataSet(
    hdf5_path = [
        os.path.join(data_path, 'test.hdf5')],
    target = target_dataset,
    task = task,
    node_feature = node_features,
    edge_feature = edge_features
)
_log.info(f'Len df train: {len(dataset_train)}')
_log.info(f'Len df valid: {len(dataset_val)}')
_log.info(f'Len df test: {len(dataset_test)}')
####################

#################### Trainer

_log.info(f'Instantiating Trainer...\n')

trainer = Trainer(
    dataset_train,
    dataset_val,
    dataset_test,
    net,
    batch_size = batch_size,
    output_dir = metrics_path
)
trainer.configure_optimizers(optimizer, lr, weight_decay)
trainer.train(nepoch = epochs, validate = True, save_model = save_model, model_path = os.path.join(exp_path, 'model.tar'))
trainer.test()

_log.info(f"Model saved at epoch {trainer.epoch_saved_model}")

#################### Metadata saving
exp_json = {}
exp_json['exp_id'] = exp_id
exp_json['input_data_path'] = input_data_path
exp_json['protein_class'] = protein_class
exp_json['target_data'] = target_data
exp_json['res_data'] = resolution_data
exp_json['target_data'] = target_data
exp_json['task'] = task
exp_json['node_features'] = [node_features]
exp_json['edge_features'] = [edge_features]
exp_json['net'] = str(net)
exp_json['batch_size'] = batch_size
exp_json['optimizer'] = str(optimizer)
exp_json['lr'] = lr
exp_json['weight_decay'] = weight_decay
exp_json['epoch'] = trainer.epoch_saved_model
exp_json['train_loss'] = np.nan
exp_json['valid_loss'] = np.nan
exp_json['test_loss'] = np.nan
exp_json['train_accuracy'] = np.nan
exp_json['valid_accuracy'] = np.nan
exp_json['test_accuracy'] = np.nan
exp_json['train_mcc'] = np.nan
exp_json['valid_mcc'] = np.nan
exp_json['test_mcc'] = np.nan
exp_json['train_f1'] = np.nan
exp_json['valid_f1'] = np.nan
exp_json['test_f1'] = np.nan
exp_json['train_rmse'] = np.nan
exp_json['valid_rmse'] = np.nan
exp_json['test_rmse'] = np.nan
exp_df = pd.DataFrame(exp_json, index=[0])

filename = Path('experiments_log.xlsx')
file_exists = filename.is_file()

with pd.ExcelWriter(
    filename,
    engine="openpyxl",
    mode="a" if file_exists else "w",
    if_sheet_exists='overlay' if file_exists else None,
) as writer:

    if file_exists:
        _log.info("Updating metadata in experiments_log.xlsx ...\n")
        startrow=writer.sheets['All'].max_row
        exp_df.to_excel(writer, sheet_name='All', startrow=startrow, index=False, header=False)
    else:
        _log.info("Creating metadata in experiments_log.xlsx ...\n")
        startrow = 0
        exp_df.to_excel(writer, sheet_name='All', startrow=startrow, index=False, header=True)

_log.info("Saved! End of the training script")

####################
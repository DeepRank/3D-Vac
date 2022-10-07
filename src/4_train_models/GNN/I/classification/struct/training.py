import h5py
import glob
import os
import sys
import torch
import pandas as pd
import logging
from deeprankcore.Trainer import Trainer
from deeprankcore.ginet import GINet
from deeprankcore.DataSet import HDF5DataSet, save_hdf5_keys

# random seed?

#################### Data to fill
data = 'pMHCI'
task = 'BA'
run_day = '22082022'
resolution = 'residue' # either 'residue' or 'atomic'
project_folder = '/Users/giuliacrocioni/Desktop/docs/eScience/projects/3D-vac/snellius_50/'
output_folder = f'{project_folder}data/{data}/features_output_folder/GNN/{resolution}/{run_day}'
hdf5_path = output_folder + '/' + resolution + '.hdf5'

targets = 'score'
cluster = 'cluster'
y = 'binary'

# features
node_features = ['bsa', 'depth', 'hb_acceptors', 'hb_donors', 'hse', 'ic', 'polarity', 'pos', 'pssm', 'sasa', 'size', 'type']
edge_features = ['coulomb', 'covalent', 'dist', 'vanderwaals']

# clusters
train_clusters = [0, 1, 2, 3, 4, 7, 9]
val_clusters = [5, 8]
test_clusters = [6]

# trainer
net = GINet
nn_task = 'class'
batch_size = 8
optimizer = torch.optim.Adam
lr = 0.01
weight_decay = 1e-05
epochs = 50
####################

# save main exp settings in a master file, with exp_id

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

_log.info("training.py has started!\n")

#################### Data summary
summary = {}
summary['entry'] = []
summary['cluster'] = []
summary['target'] = []
summary['phase'] = []

# '/Users/giuliacrocioni/remote_snellius/data/pMHCI/features_output_folder/GNN/residue/13072022/residue.hdf5'
with h5py.File(hdf5_path, 'r') as hdf5:

    for mol in hdf5.keys():
        cluster_value = float(hdf5[mol][targets][cluster][()])
        target_value = float(hdf5[mol][targets][y][()])

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
    os.path.join(exp_path, 'summary_data.hdf5'),
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
        _log.info(f'\t\t\t- Class 0: {len(df_summ[df_summ.cluster == cl].target == 0)} samples, {round(100*len(df_summ[df_summ.cluster == cl].target == 0)/len(df_summ))}%')
        _log.info(f'\t\t\t- Class 1: {len(df_summ[df_summ.cluster == cl].target == 1)} samples, {round(100*len(df_summ[df_summ.cluster == cl].target == 1)/len(df_summ))}%')
    else:
        _log.info(f'Cluster {int(cl)} not present!')

save_hdf5_keys(hdf5_path, df_summ[df_summ.phase == 'train'].entry.to_list(), os.path.join(exp_path, 'train.hdf5'), hardcopy = True)
save_hdf5_keys(hdf5_path, df_summ[df_summ.phase == 'valid'].entry.to_list(), os.path.join(exp_path, 'valid.hdf5'), hardcopy = True)
save_hdf5_keys(hdf5_path, df_summ[df_summ.phase == 'test'].entry.to_list(), os.path.join(exp_path, 'test.hdf5'), hardcopy = True)
####################

#################### HDF5DataSet

_log.info(f'HDF5DataSet loading...\n')

# to change: pass in only list of keys
dataset_train = HDF5DataSet(
    hdf5_path = os.path.join(exp_path, 'train.hdf5'),
    target = y,
    node_feature = node_features,
    edge_feature = edge_features
)
dataset_val = HDF5DataSet(
    hdf5_path = os.path.join(exp_path, 'valid.hdf5'),
    target = y,
    node_feature = node_features,
    edge_feature = edge_features
)
dataset_test = HDF5DataSet(
    hdf5_path = os.path.join(exp_path, 'test.hdf5'),
    target = y,
    node_feature = node_features,
    edge_feature = edge_features
)
####################

#################### Trainer

_log.info(f'Instantiating Trainer...\n')

trainer = Trainer(
    dataset_train,
    dataset_val,
    dataset_test,
    net,
    task = nn_task,
    batch_size = batch_size,
    output_dir = exp_path
)
trainer.configure_optimizers(optimizer, lr, weight_decay)
trainer.train(nepoch = epochs, validate = True, save_model = 'best')
trainer.test()

# Remember to install pytables
_log.info("Saving model...")
trainer.save_model(filename = os.path.join(exp_path, 'model.tar'))
_log.info("Done!")

####################
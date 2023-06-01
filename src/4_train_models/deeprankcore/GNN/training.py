import os
import sys
import glob
from pathlib import Path
from datetime import datetime
import logging
import cProfile, pstats, io
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef)
import torch
from deeprankcore.trainer import Trainer
from deeprankcore.utils.exporters import HDF5OutputExporter
from deeprankcore.dataset import GraphDataset
from deeprankcore.neuralnets.gnn.naive_gnn import NaiveNetwork
from pmhc_gnn import PMHCI_Network01, PMHCI_Network02, PMHCI_Network03


# rerun training 100k 230530 (only graphs) and new transformation thing for
# DONE shuffled 2816985 (net1)

# ONGOING shuffled 2820689 (net3)
# TODO cl_peptide
# TODO cl_peptide2
# TODO cl_allele
# TODO allele_type?

# initialize
starttime = datetime.now()
torch.manual_seed(22)

#################### To fill
# Input data
# run_day_data = '230515' # 100k and 692 data points, grids + graphs
run_day_data = '230530' # 100k and 692 data points, only graphs
# Paths
protein_class = 'I'
target_data = 'BA'
resolution_data = 'residue' # either 'residue' or 'atomic'
# project_folder = '/home/ccrocion/snellius_data_sample' # local resized df path
project_folder = '/projects/0/einf2380'
folder_data = f'{project_folder}/data/pMHC{protein_class}/features_output_folder/deeprankcore/{resolution_data}/{run_day_data}'
input_data_path = glob.glob(os.path.join(folder_data, '*.hdf5'))
# Experiment naming
exp_basepath = f'{project_folder}/data/pMHC{protein_class}/trained_models/deeprankcore/experiments/'
exp_name = 'exp_100k_std_transf_bs64_net3_'
exp_date = True # bool
exp_suffix = ''
# Target/s
target_group = 'target_values'
target_dataset = 'binary'
task = 'classif'
features_transform = {'bsa': {'Transformation': lambda t: np.log(t+1), 'Standardization': True},
               'res_depth': {'Transformation': lambda t: np.log(t+1), 'Standardization': True},
               'info_content': {'Transformation': lambda t: np.log(t+1), 'Standardization': True},
               'sasa': {'Transformation': lambda t: np.sqrt(t), 'Standardization': True},
               'electrostatic': {'Transformation': lambda t: np.cbrt(t), 'Standardization': True},
               'vanderwaals': {'Transformation': lambda t: np.cbrt(t), 'Standardization': True},
               'res_size': {'Transformation': None, 'Standardization': True},
               'res_charge': {'Transformation': None, 'Standardization': True},
               'hb_donors': {'Transformation': None, 'Standardization': True},
               'hb_acceptors': {'Transformation': None, 'Standardization': True},
               'hse': {'Transformation': None, 'Standardization': True},
               'irc_nonpolar_negative': {'Transformation': None, 'Standardization': True},
               'irc_nonpolar_nonpolar': {'Transformation': None, 'Standardization': True},
               'irc_nonpolar_polar': {'Transformation': None, 'Standardization': True},
               'irc_nonpolar_positive': {'Transformation': None, 'Standardization': True},
               'irc_polar_polar': {'Transformation': None, 'Standardization': True},
               'irc_polar_positive': {'Transformation': None, 'Standardization': True},
               'irc_total': {'Transformation': None, 'Standardization': True},
               'irc_negative_positive': {'Transformation': None, 'Standardization': True},
               'irc_positive_positive': {'Transformation': None, 'Standardization': True},
               'irc_polar_negative': {'Transformation': None, 'Standardization': True},
               'irc_negative_negative': {'Transformation': None, 'Standardization': True},
               'res_mass': {'Transformation': None, 'Standardization': True},
               'res_pI': {'Transformation': None, 'Standardization': True},
               'distance': {'Transformation': None, 'Standardization': True},
               'pssm': {'Transformation': None, 'Standardization': True}}
# Clusters
# If cluster_dataset is None, sets are randomly splitted
cluster_dataset = None # 'cl_peptide' # 'cl_peptide2' # 'cl_allele' # 'allele_type' # None
cluster_dataset_type = None # None # 'string'
test_clusters = ['C']
# Dataset
node_features = "all"
edge_features = "all"
# Trainer
net = PMHCI_Network03
batch_size = 64
optimizer = torch.optim.Adam
lr = 1e-3
weight_decay = 0
epochs = 70
class_weights = False # weighted loss function
cuda = True
ngpu = 1
num_workers = 16
train_profiling = False
check_integrity = True
# early stopping
earlystop_patience = 20
earlystop_maxgap = 0.06
min_epoch = 45
####################


#################### Folders and logger
# Outputs folder
exp_id = exp_name + '0'
if os.path.exists(exp_basepath):
    exp_list = [f for f in os.listdir(exp_basepath) if f.lower().startswith(exp_name.lower())]
    if len(exp_list) > 0:
        last_id = max([int(w[len(exp_name):].split('_')[0]) for w in exp_list])
        exp_id = exp_name + str(last_id + 1)
exp_path = os.path.join(exp_basepath, exp_id)
if exp_date:
    today = starttime.strftime('%y%m%d')
    exp_path += '_' + today
if exp_suffix:
    exp_path += '_' + exp_suffix
os.makedirs(exp_path)

data_path = os.path.join(exp_path, 'data')
output_path = os.path.join(exp_path, 'output')
img_path = os.path.join(exp_path, 'images')
os.makedirs(data_path)
os.makedirs(output_path)
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

if __name__ == "__main__":
    _log.info(f'Created folder {exp_path}\n')

    _log.info("training.py has started!\n")

    #################### Data summary
    summary = {}
    summary['entry'] = []
    summary['target'] = []

    if cluster_dataset is not None:
        summary['cluster'] = []

    for fname in input_data_path:
        try:
            with h5py.File(fname, 'r') as hdf5:
                for mol in hdf5.keys():
                    target_value = float(hdf5[mol][target_group][target_dataset][()])
                    summary['entry'].append(mol)
                    summary['target'].append(target_value)

                    if cluster_dataset is not None:
                        if cluster_dataset_type == 'string':
                            cluster_value = hdf5[mol][target_group][cluster_dataset].asstr()[()]
                        else:
                            cluster_value = float(hdf5[mol][target_group][cluster_dataset][()])

                        summary['cluster'].append(cluster_value)

        except Exception as e:
            _log.error(e)
            _log.info(f'Error in opening {fname}, please check the file.')
    
    df_summ = pd.DataFrame(data=summary)

    if cluster_dataset is None:
        # random split
        df_train, df_test = train_test_split(df_summ, test_size=0.1, stratify=df_summ.target, random_state=42)
        df_train, df_valid = train_test_split(df_train, test_size=0.2, stratify=df_train.target, random_state=42)
    else:
        # use cluster for test, random split for train and valid
        df_test = df_summ[df_summ.cluster.isin(test_clusters)]
        df_train = df_summ[~df_summ.cluster.isin(test_clusters)]
        df_train, df_valid = train_test_split(df_train, test_size=0.2, stratify=df_train.target, random_state=42)

    df_summ['phase'] = ['test' if entry in df_test.entry.values else 'valid' if entry in df_valid.entry.values else 'train' for entry in df_summ.entry]

    df_summ.to_hdf(
        os.path.join(output_path, 'summary_data.hdf5'),
        key='summary',
        mode='w')

    _log.info(f'Data statistics:\n')
    _log.info(f'Total samples: {len(df_summ)}\n')
    if cluster_dataset is not None:
        _log.info(f'Clustering on Dataset: {cluster_dataset}.\n')
    _log.info(f'Training set: {len(df_train)} samples, {round(100*len(df_train)/len(df_summ))}%')
    _log.info(f'\t- Class 0: {len(df_train[df_train.target == 0])} samples, {round(100*len(df_train[df_train.target == 0])/len(df_train))}%')
    _log.info(f'\t- Class 1: {len(df_train[df_train.target == 1])} samples, {round(100*len(df_train[df_train.target == 1])/len(df_train))}%')
    if cluster_dataset is not None:
        _log.info(f'Clusters present: {df_train.cluster.unique()}\n')
    _log.info(f'Validation set: {len(df_valid)} samples, {round(100*len(df_valid)/len(df_summ))}%')
    _log.info(f'\t- Class 0: {len(df_valid[df_valid.target == 0])} samples, {round(100*len(df_valid[df_valid.target == 0])/len(df_valid))}%')
    _log.info(f'\t- Class 1: {len(df_valid[df_valid.target == 1])} samples, {round(100*len(df_valid[df_valid.target == 1])/len(df_valid))}%')
    if cluster_dataset is not None:
        _log.info(f'Clusters present: {df_valid.cluster.unique()}\n')
    _log.info(f'Testing set: {len(df_test)} samples, {round(100*len(df_test)/len(df_summ))}%')
    _log.info(f'\t- Class 0: {len(df_test[df_test.target == 0])} samples, {round(100*len(df_test[df_test.target == 0])/len(df_test))}%')
    _log.info(f'\t- Class 1: {len(df_test[df_test.target == 1])} samples, {round(100*len(df_test[df_test.target == 1])/len(df_test))}%')
    if cluster_dataset is not None:
        _log.info(f'Clusters present: {df_test.cluster.unique()}\n')

    #################### GraphDataset

    _log.info(f'HDF5DataSet loading...\n')
    dataset_train = GraphDataset(
        hdf5_path = input_data_path,
        subset = list(df_train.entry),
        target = target_dataset,
        task = task,
        node_features = node_features,
        edge_features = edge_features,
        features_transform = features_transform,
        check_integrity = check_integrity
    )

    dataset_val = GraphDataset(
        hdf5_path = input_data_path,
        subset = list(df_valid.entry),
        target = target_dataset,
        task = task,
        node_features = node_features,
        edge_features = edge_features,
        train = False,
        dataset_train = dataset_train,
        check_integrity = check_integrity
    )
    dataset_test = GraphDataset(
        hdf5_path = input_data_path,
        subset = list(df_test.entry),
        target = target_dataset,
        task = task,
        node_features = node_features,
        edge_features = edge_features,
        train = False,
        dataset_train = dataset_train,
        check_integrity = check_integrity
    )
    _log.info(f'Len df train: {len(dataset_train)}')
    _log.info(f'Len df valid: {len(dataset_val)}')
    _log.info(f'Len df test: {len(dataset_test)}')
    _log.info(f'Node features: {dataset_train.node_features}')
    _log.info(f'Edge features: {dataset_train.edge_features}')
    _log.info(f'Target: {dataset_train.target}')
    _log.info(f'Task: {dataset_train.task}')
    _log.info(f'Standardize: {features_transform}')
    ####################

    #################### Trainer

    _log.info(f'Instantiating Trainer...\n')

    trainer = Trainer(
        net,
        dataset_train,
        dataset_val,
        dataset_test,
        class_weights = class_weights,
        cuda = cuda,
        ngpu = ngpu,
        output_exporters = [HDF5OutputExporter(output_path)]
    )
    trainer.configure_optimizers(optimizer, lr, weight_decay)

    if train_profiling:
        _log.info(f"Number of workers set to {num_workers}.")
        pr = cProfile.Profile()
        pr.enable()
        trainer.train(
            nepoch = epochs,
            batch_size = batch_size,
            validate = True,
            num_workers = num_workers,
            filename = os.path.join(exp_path, 'model.pth.tar'))
        pr.disable()

        _log.info(f"Batch size set to {trainer.batch_size_train}.")

        s_tot = io.StringIO()
        s_cum = io.StringIO()
        s_n = io.StringIO()

        ps_tot = pstats.Stats(pr, stream=s_tot).strip_dirs().sort_stats('tottime').print_stats()
        ps_cum = pstats.Stats(pr, stream=s_cum).strip_dirs().sort_stats('cumtime').print_stats()
        ps_n = pstats.Stats(pr, stream=s_n).strip_dirs().sort_stats('ncalls').print_stats()

        # Save it into disk
        with open(os.path.join(exp_path, 'cProfile_tottime.txt'), 'w+') as f:
            f.write(s_tot.getvalue())
        with open(os.path.join(exp_path, 'cProfile_cumtime.txt'), 'w+') as f:
            f.write(s_cum.getvalue())
        with open(os.path.join(exp_path, 'cProfile_ncalls.txt'), 'w+') as f:
            f.write(s_n.getvalue())
        
        _log.info(f"Train ended, complexity profiled.")
    else:
        _log.info(f"Number of workers set to {num_workers}.")
        _log.info(f"Class weight: {trainer.class_weights}.")
        _log.info(f"Learning rate set to {trainer.lr}.")
        _log.info(f"Max number of epochs set to {epochs}.")
        _log.info(f"earlystop_patience set to {earlystop_patience}.")
        _log.info(f"earlystop_maxgap set to {earlystop_maxgap}.")
        _log.info(f"min_epoch set to {min_epoch}.")

        trainer.train(
            nepoch = epochs,
            batch_size = batch_size,
            earlystop_patience = earlystop_patience,
            earlystop_maxgap = earlystop_maxgap,
            min_epoch = min_epoch,
            validate = True,
            num_workers = num_workers,
            filename = os.path.join(exp_path, 'model.pth.tar'))
        _log.info(f"Batch size set to {trainer.batch_size_train}.")
        trainer.test(batch_size = batch_size, num_workers = num_workers)

        epoch = trainer.epoch_saved_model
        _log.info(f"Model saved at epoch {epoch}")
        pytorch_total_params = sum(p.numel() for p in trainer.model.parameters())
        _log.info(f'Total # of parameters: {pytorch_total_params}')
        pytorch_trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        _log.info(f'Total # of trainable parameters: {pytorch_trainable_params}')

        #################### Metadata saving
        exp_json = {}

        ## store input settings
        exp_json['exp_id'] = exp_id
        exp_json['exp_fullname'] = exp_path.split('/')[-1]
        exp_json['exp_path'] = exp_path
        exp_json['start_time'] = starttime.strftime("%d/%b/%Y_%H:%M:%S")
        exp_json['end_time'] = '_' #placeholder to keep location
        exp_json['input_data_path'] = [input_data_path]
        exp_json['protein_class'] = protein_class
        exp_json['target_data'] = target_data
        exp_json['resolution'] = resolution_data
        exp_json['target_data'] = target_data
        exp_json['task'] = task
        exp_json['node_features'] = 'all'
        exp_json['edge_features'] = 'all'
        exp_json['net'] = str(net)
        exp_json['optimizer'] = str(optimizer)
        exp_json['max_epochs'] = epochs
        exp_json['batch_size'] = batch_size
        exp_json['lr'] = lr
        exp_json['weight_decay'] = weight_decay
        exp_json['save_state'] = 'best'
        exp_json['train_datapoints'] = len(df_train)
        exp_json['val_datapoints'] = len(df_valid)
        exp_json['test_datapoints'] = len(df_test)
        exp_json['total_datapoints'] = len(df_summ)
        if cluster_dataset is not None:
            # exp_json['train_clusters'] = [train_clusters]
            # exp_json['val_clusters'] = [val_clusters]
            exp_json['test_clusters'] = [test_clusters]

        ## load output and retrieve metrics
        exp_json['saved_epoch'] = epoch
        exp_json['last_epoch'] = epochs # adjust if/when we add an early stop

        output_train = pd.read_hdf(os.path.join(output_path, 'output_exporter.hdf5'), key='training')
        output_test = pd.read_hdf(os.path.join(output_path, 'output_exporter.hdf5'), key='testing')
        output_df = pd.concat([output_train, output_test])

        d = {'thr': [], 'precision': [], 'recall': [], 'accuracy': [], 'f1': [], 'mcc': [], 'auc': [], 'aucpr': [], 'phase': []}
        thr_df = pd.DataFrame(data=d)
        df_epoch = output_df[(output_df.epoch == epoch) | ((output_df.epoch == 0) & (output_df.phase == 'testing'))]

        for phase in ['training', 'validation', 'testing']:
            df_epoch_phase = df_epoch[(df_epoch.phase == phase)]
            y_true = df_epoch_phase.target
            y_score = np.array(df_epoch_phase.output.values.tolist())[:, 1]

            thrs = np.linspace(0,1,100)
            precision = []
            recall = []
            accuracy = []
            f1 = []
            mcc = []
            
            for thr in thrs:
                y_pred = (y_score > thr)*1
                precision.append(precision_score(y_true, y_pred, zero_division=0))
                recall.append(recall_score(y_true, y_pred, zero_division=0))
                accuracy.append(accuracy_score(y_true, y_pred))
                f1.append(f1_score(y_true, y_pred, zero_division=0))
                mcc.append(matthews_corrcoef(y_true, y_pred))

            fpr_roc, tpr_roc, thr_roc = roc_curve(y_true, y_score)
            auc_score = auc(fpr_roc, tpr_roc)
            aucpr = average_precision_score(y_true, y_score)

            phase_df = pd.DataFrame({'thr': thrs, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1': f1, 'mcc': mcc, 'auc': auc_score, 'aucpr': aucpr, 'phase': phase})
            thr_df = pd.concat([thr_df, phase_df], ignore_index=True)

        # find max mcc of test set
        test_df = thr_df.loc[thr_df.phase == 'testing']
        test_mcc_idxmax = test_df.mcc.idxmax()
        if thr_df.loc[test_mcc_idxmax].mcc > 0:
            sel_thr = thr_df.loc[test_mcc_idxmax].thr
        # use max mcc of all data if max of test set is 0 (usually only on small local test experiments)
        else:
            mcc_idxmax = thr_df.mcc.idxmax()
            sel_thr = thr_df.loc[mcc_idxmax].thr
            _log.info("WARNING: Maximum mcc of test set is 0. Instead, maximum mcc of all data will be used for determining optimal threshold.\n")

        ## store output
        exp_json['training_loss'] = output_df[(output_df.epoch == epoch) & (output_df.phase == 'training')].loss.mean()
        exp_json['validation_loss'] = output_df[(output_df.epoch == epoch) & (output_df.phase == 'validation')].loss.mean()
        exp_json['testing_loss'] = output_df[(output_df.epoch == epoch) & (output_df.phase == 'testing')].loss.mean()
        for score in ['mcc', 'auc', 'aucpr', 'f1', 'accuracy', 'precision', 'recall']:
            for phase in ['training', 'validation', 'testing']:
                exp_json[f'{phase}_{score}'] = round(float(thr_df[(thr_df.thr == sel_thr) & (thr_df.phase == phase)][score]), 3)


        # Output to excel file
        exp_json['end_time'] = datetime.now().strftime("%d/%b/%Y_%H:%M:%S")
        exp_df = pd.DataFrame(exp_json, index=[0])
        filename = Path(exp_basepath + '_experiments_log.xlsx')
        file_exists = filename.is_file()

        with pd.ExcelWriter(
            filename,
            engine="openpyxl",
            mode="a" if file_exists else "w",
            if_sheet_exists='overlay' if file_exists else None,
        ) as writer:

            if file_exists:
                _log.info("Updating metadata in experiments_log.xlsx ...\n")
                old_df = pd.read_excel(filename)
                exp_df = pd.concat([exp_df, old_df]) # newest experiment on top
            else:
                _log.info("Creating metadata in experiments_log.xlsx ...\n")
            exp_df.to_excel(writer, sheet_name='All', index=False, header=True)

        _log.info("Saved! End of the training script")

    ####################

#!/usr/bin/env python
# Original author: Francesco Ambrosetti, modified by Daniil Lepikhov

import h5py
import random
import os
import glob
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from joblib import Parallel, delayed
import numpy as np
import random
import pandas as pd
import pickle
import tqdm
from collections import Counter

arg_parser = argparse.ArgumentParser(
    epilog="""
    python ./split_h5.py \
    --csv-file ../../../data/external/processed/all_hla_j_4.csv \
    --features-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative/ \
    --output-path /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/h5_test \
    --cluster \ #perform a clustered training and test
    --cluster-column cluster_set_10 \ #column of the csv file containing cluster number
    --train-clusters 0 1 2 4 5 6 7 8 9 \ # cluster numbers (from --cluster-column) used for training
    --test-clusters 3 
    """,
    description=
    """
    Combines every h5 files provided in the --features-path argument
    into a list. The list can be either shuffled in cross-validation sets or clustered using
    leave one group out (--cluster argument). The script can split based on clusters used for train and test 
    (--cluster, --train-clusters --test-clusters arguments and --cluster-column to a specific 
    cluster_set_{n} column). Note that when using specific clusters, check if the cluster number is indexed.
    Which means if we want cluster 4 as the cluster used for test, we would have to set --test-clusters to 3
    if it is indexed.
    For the shuffling, the list is randomly shuffled n times (number of folds for the X-validation). 
    For the clustering without specific clusters for train and test, the list is separated based on 
    clusters provided in the --cluster-column argument (see description of argument for more details).
    Generated splits are dumped into the --output-path and then used for training the CNN. 
    Providing a --trainval-csv argument, train and validation splits are going to be generated
    from this file. --test-csv will be used as the test.
    ID column of the --csv-file are used as filters to select cases from --features-path.
    """
)
arg_parser.add_argument("--features-path", "-f",
    help="Path to the folder containing all h5 features generated by the 4/generate_features.py file. Default to: \
    /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/"
)

arg_parser.add_argument("--cluster-column", "-C",
    help="""
    Column in the csv file containing peptides' cluster annotations.
    Should be formated like cluster_set_{n} and provided with the --cluster argument. Default column 'cluster'. 
    Example: --cluster-column cluster_set_10
    """,
    default="cluster"
)

arg_parser.add_argument("--train-clusters", "-T",
    help="""
    Provide one or several clusters (as rows from the --cluster-column) used for training. Use with --cluster.
    Required with --test-clusters. Example: --train-clusters 0 1 2 3 4 5 6
    """,
    nargs="+",
    type=int
)
arg_parser.add_argument("--test-clusters", "-V",
    help="""
    Provide one or several clusters used for testing. Use with --cluster.
    Required with --train-clusters.
    """,
    nargs="+",
    type=int
)
arg_parser.add_argument("--cluster", "-c",
    help = "If provided, creates --n-fold train, validation, test splits for a leave-one-group-out cross validation.",
    default = False,
    action = "store_true",
)

arg_parser.add_argument("--n-fold", "-n",
    help = """
    Number of folds for the cross-validation (only when the training and test is not clustered).
    Default 10. 
    """,
    default=10,
    type=int
)

arg_parser.add_argument("--single-split", "-s",
    help = "Do not make multiple splits for cross validation but just a single train-test-validate with test being value 0 in 'cluster' column.",
    default = False,
    action = "store_true",
)

arg_parser.add_argument("--train-val-split", "-S",
    help = "The ratio for the train and validation set, two number seperated with a '-'. Eg. 85-15 (should add up to 100)",
    default = '85-15',
    type=str,
)

arg_parser.add_argument("--parallel", "-p",
    help = "runs the process in parallel on a slurm cluster",
    default = 1,
    type = int,
)

arg_parser.add_argument("--csv-file", "-d",
    help= "Name of db1. Needed only for clustered split.",
    default=None,
)
arg_parser.add_argument("--output-path", "-o",
    help="Destination path for the generated splits. For clustered will be in --output-path/clustered and in \
    --output-path/shuffled for shuffled. Default: \
    /projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/splits/",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/splits"
)
arg_parser.add_argument("--trainval-csv",
    help="Path to the CSV containing train and validation cases. It is not required, can be used in the case of train and validation cases are in a separate csv.",
    default=None,
)
arg_parser.add_argument("--test-csv",
    help="Path to the CSV containing test cases. Should be used with --trainval-csv",
    default=None
)

a = arg_parser.parse_args()

# Change the combined_path and output_h5:
output_h5_path = a.output_path

def read_caseIDS_from_hfile(hfile, ids):
    """Read a hdf5 file, returns 2 lists:
    - caseIDS: the name of the model (starting with BA_ or EL_)
    - label: the classification task target value (BIN_CLASS property)

    arguments:
    f5_file = path to the hdf5 file"""
    caseIDs = []
    labels = []
    with h5py.File(hfile, 'r') as h5:
        h5cases = list(h5.keys())
        caseIDs = [case_id for case_id in h5cases if case_id in ids]
        for case in caseIDs:
            try:
                labels.append(h5[case]["targets"]["BIN_CLASS"][()])
            except:
                print('#########')
                print(f'Problem occurred with case {case}')
                #raise Exception(f'Problem occurred with case {case}')
    return caseIDs, labels

def h5_symlinks(input_path, ids):
    """Combines HDF5 files from the input_path folder
    into a dictionaries of symlinks which then will be splited.
    Return values:
    - symlinks: symlink dictionary where the key is the model name (caseID)
    and the value is the HDF5 file where the caseID is stored.
    - labels: model's labels from every HDF5 files ordered in the same order
    as the keys in the symlinks variable.
    """
    hfiles = glob.glob(os.path.join(input_path,'*.hdf5'))
    symlinks = {}
    # labels = []
    
    caseIDs, case_labels = zip(*Parallel(verbose=True, n_jobs=n_cores)(delayed(read_caseIDS_from_hfile)(hfile, ids) for hfile in hfiles))
    for i, hfile in enumerate(hfiles):
        for caseID in caseIDs[i]:
            symlinks[caseID] = hfile
    # flatten the list of labels
    labels = [lab for sub_labels in case_labels for lab in sub_labels]
    
    return symlinks, labels


def save_train_valid(train_idx, val_idx, test_idx, symlinks, features_path, path_out_folder,
    train_f, valid_f, test_f
):
    """Creates the train, valid and test HDF5 files from the symlinks
    with the caseIDS provided.

    Arguments:
    train_idx = caseIDs from symlinks for the training set (from split_train())
    val_idx = casedIDs from symlinks for the validation set (from split_train())
    test_idx = casedIDs from symlinks for the test set (from split_train())

    out_folder = path to the output folder (output_h5_path)
    train_f, valid_f, test_f: filename of train, valid and test
    """
    # Create new hd5f files for the training and validation
    train_h5 = h5py.File(os.path.join(path_out_folder, train_f), 'w')
    val_h5 = h5py.File(os.path.join(path_out_folder, valid_f), 'w')
    test_h5 = h5py.File(os.path.join(path_out_folder, test_f), 'w')
    
    org_h5s = glob.glob(os.path.join(features_path,'*.hdf5'))

    print("### Creating train.hdf5 file ###")
    for org_h5 in tqdm.tqdm(org_h5s):
        entries = [] 
        for i, value_id in enumerate(train_idx):
            try:
                if symlinks[value_id] == org_h5:
                    entries.append(value_id)
                    train_idx = np.delete(train_idx, np.where(train_idx ==i))
            except KeyError as ie:
                print(ie)
                print(f'train id not found in hdf5 files {value_id}') 
        symlink_in_h5(entries, org_h5, train_h5)

    print("### Creating validation.hdf5 file ###") 
    for org_h5 in tqdm.tqdm(org_h5s):
        entries = [] 
        for i, value_id in enumerate(val_idx):
            try:
                if symlinks[value_id] == org_h5:
                    entries.append(value_id)
                    val_idx = np.delete(val_idx, np.where(val_idx ==i))
            except KeyError as ie:
                print(ie)
                print(f'validation id not found in hdf5 files {value_id}') 
        symlink_in_h5(entries, org_h5, val_h5)

    print("### Creating test.hdf5 file ###") 
    for org_h5 in tqdm.tqdm(org_h5s):
        entries = [] 
        for i, value_id in enumerate(test_idx):
            try:
                if symlinks[value_id] == org_h5:
                    entries.append(value_id)
                    test_idx = np.delete(test_idx, np.where(test_idx ==i))
            except KeyError as ie:
                print(ie)
                print(f'test id not found in hdf5 files {value_id}') 
        symlink_in_h5(entries, org_h5, test_h5) 
            
    train_h5.close()
    val_h5.close()
    test_h5.close()


def symlink_in_h5(indices, f1_path, f2):
    """Copy the selected keys (idx) of one hdf5 (f1) file into
     another hdf5 file (f2)
     idx = list of keys to copy
     f1 = handle of the first hdf5 file
     f2 = handle of the the second hdf5 file"""
    with h5py.File(f1_path, "r") as f1:
        
        for idx in indices:
            group_path = f1[idx].name
            f2[group_path] = h5py.ExternalLink(f1_path, group_path)
        
def split_one_fold(df, test_group, train_split):
    test_mask = (df["ID"].isin(all_cases)) & (df[a.cluster_column].isin(test_group))
    test_cases = df[test_mask]["ID"]
    print(f'test cases: {test_cases.shape}')
    not_test_cases = np.array(df[test_mask == False]["ID"])
    print(f'not test case {not_test_cases.shape}')
    random.shuffle(not_test_cases)
    ds_l = not_test_cases.shape[0]
    train_cases, validation_cases = np.split(
        not_test_cases, 
        [int((train_split/100)*ds_l)]
    )
    print(f'train shape: {train_cases.shape}')
    print(f'validation shape: {validation_cases.shape}')

    print(f"### SAVING SPLITS FOR CLUSTER {i}")
    save_train_valid(train_cases, validation_cases, test_cases, symlinks, a.features_path, output_h5_path,
        f"clustered/{i}/train.hdf5",
        f"clustered/{i}/valid.hdf5",
        f"clustered/{i}/test.hdf5"
    )

if __name__ == '__main__':
    print(type(a.test_csv) == str)
    
    if a.parallel:
        n_cores = a.parallel
    else:
        n_cores = 1
    try:
        train_split = int(a.train_val_split.split('-')[0])
        val_split = int(a.train_val_split.split('-')[1])
    except ValueError as ve:
        print(f'{ve}\ntrain-val-split argument should contain only integers without spaces')
    if train_split + val_split != 100:
        raise argparse.ArgumentTypeError('train-val-split should add up to 100')
        
    # Combine the h5 files using the csv as a filter. Use one of the csv provided:
    if type(a.csv_file) == str:
        df = pd.read_csv(a.csv_file)
        all_ids = df.ID.tolist()
    elif type(a.trainval_csv) == str and type(a.test_csv) == str:
        train_df = pd.read_csv(a.trainval_csv)
        test_df = pd.read_csv(a.test_csv)
        all_ids = train_df.ID.tolist() + test_df.ID.tolist()

    #Fill in the nan values (empty fileds belonging to trashbin clusters) with n_clusters + 1 
    # if a.cluster:
        # try:
        #     trashbin_cluster = len(a.train_clusters) + len(a.test_clusters) + 1
        # except TypeError:
        #     try:
        #         trashbin_cluster = int(a.cluster_column.split('_')[-1])
        #     except:
        #         raise Exception('Something went wrong in defining trashbin cluster')
                
        # df.fillna(trashbin_cluster,inplace=True)

    symlinks, labels = h5_symlinks(a.features_path, all_ids)

    n_splits=a.n_fold

    #Make the output directories if they are not present already
    if not a.single_split:
        split_type = ('shuffled', 'clustered')[a.cluster]
        for split in range(0,n_splits):
            if not os.path.isdir(output_h5_path + f"/{split_type}/{split}"):
                os.makedirs(output_h5_path + f"/{split_type}/{split}")
    else:
        split_type = ('shuffled', 'clustered')[a.cluster]
        if not os.path.isdir(output_h5_path + f"/{split_type}/"):
            os.makedirs(output_h5_path + f"/{split_type}/")
            
    if a.cluster == False and not a.trainval_csv and not a.testval_csv:
        all_cases = np.array(list(symlinks.keys()))
        labels = np.array(labels)
        indices = np.array(range(len(labels)))
        random.shuffle(indices)

        kfold = StratifiedKFold(n_splits = n_splits)
        i = 0
        for training_idx, test_idx in kfold.split(indices, labels[indices]):
            training_idx, validation_idx = train_test_split(training_idx, test_size=val_split/100, stratify=labels[training_idx])
            
            tr = all_cases[training_idx]
            va = all_cases[validation_idx]
            te = all_cases[test_idx]
            # create training and validation hdf5 files
            print(f"### SAVING SPLITS for {i} ###")
            save_train_valid(tr, va, te, symlinks, a.features_path, output_h5_path,
                f"shuffled/{i}/train.hdf5", 
                f"shuffled/{i}/valid.hdf5", 
                f"shuffled/{i}/test.hdf5")
            i+=1
    elif type(a.trainval_csv) == str and type(a.test_csv) == str:
        train_val_cases_df = pd.read_csv(a.trainval_csv)
        test_cases_df = pd.read_csv(a.test_csv)

        ds_l = train_val_cases_df.shape[0]
        train_val_cases = np.array(train_val_cases_df.ID.tolist())
        random.shuffle(train_val_cases)

        test_cases = test_cases_df.ID

        train_cases, validation_cases = np.split(
            train_val_cases, 
            [int((train_split/100)*ds_l)]
        )
        print(f"""
        ### SAVING TRAINING SPLITS FROM TRAIN, VALIDATION CSV FILE {a.trainval_csv.split("/")[-1]} AND
        ### TESTING SPLITS FROM CSV FILE {a.test_csv.split("/")[-1]} FROM CLUSTER COLUMN 
        
        """)
        save_train_valid(train_cases, validation_cases, test_cases, symlinks, a.features_path, output_h5_path,
            f"train.hdf5",
            f"valid.hdf5",
            f"test.hdf5"
        )

    elif a.cluster:
        groups = [int(x) for x in set(df[a.cluster_column])]
        all_cases = list(symlinks.keys())
        if a.test_clusters:
            split_one_fold(df, a.test_clusters, train_split)
            
            test_mask = (df["ID"].isin(all_cases)) & (df[a.cluster_column].isin(a.test_clusters))
            train_val_mask = (df["ID"].isin(all_cases)) & (~df[a.cluster_column].isin(a.test_clusters))

            test_cases = df[test_mask]["ID"]
            train_val_cases = np.array(df[train_val_mask]["ID"])

            ds_l = train_val_cases.shape[0]
            train_cases, validation_cases = np.split(
                train_val_cases, 
                [int((train_split/100)*ds_l)]
            )

            print(f"""
            ### SAVING TEST SPLITS FROM CLUSTERS {a.test_clusters}
            
            """)
            
            save_train_valid(train_cases, validation_cases, test_cases, symlinks, a.features_path, output_h5_path,
                f"clustered/train.hdf5",
                f"clustered/valid.hdf5",
                f"clustered/test.hdf5"
            )

        else: #perform normal clustering
            jobs = min([len(groups), n_cores])
            Parallel(verbose=True, n_jobs=jobs)(delayed(split_one_fold)(df, test_group, train_split) for test_group in groups)

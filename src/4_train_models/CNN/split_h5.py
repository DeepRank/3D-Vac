#!/usr/bin/env python
# Original author: Francesco Ambrosetti, modified by Daniil Lepikhov

import h5py
import random
import os
import glob
import argparse
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import random
import pandas as pd

arg_parser = argparse.ArgumentParser(
    description="This script combines every h5 files provided in the --features-path argument\
    into a list. This list is randomly shuffled or clustered 10 times (for 10 fold xvalidation) \
    using clusters provided in --csv-file cluster colum. \
    Generated splits are dumped into the --output-path and then used for training the CNN."
)
arg_parser.add_argument("--features-path", "-f",
    help="Path to the folder containing all h5 features generated by the 4/generate_features.py file. Default to: \
    /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/"
)

arg_parser.add_argument("--cluster", "-c",
    help = "If provided, creates 10 train, validation, test splits for a leave one group out cross validation.",
    default = False,
    action = "store_true",
)

arg_parser.add_argument("--csv-file", "-d",
    help= "Name of db1. Needed only for clustered split.",
    default="../../../data/external/processed/BA_pMHCI.csv",
)
arg_parser.add_argument("--output-path", "-o",
    help="Destination path for the generated splits. For clustered will be in --output-path/clustered and in \
    --output-path/shuffled for shuffled. Default: \
    /projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/splits/",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/splits"
)

a = arg_parser.parse_args()

# Change the combined_path and output_h5:
output_h5_path = a.output_path

def h5_symlinks(input_path):
    """Combines HDF5 files from the input_path folder
    into a dictionaries of symlinks which then will be splited.
    Return values:
    - symlinks: symlink dictionary where the key is the model name (caseID)
    and the value is the HDF5 file where the caseID is stored.
    - labels: model's labels from every HDF5 files ordered in the same order
    as the keys in the symlinks variable.
    """
    hfiles = glob.glob(f"{input_path}*.hdf5")
    symlinks = {}
    labels = []
    for hfile in hfiles:
        try:
            caseIDs, case_labels = read_caseIDS_from_hfile(hfile)
        except:
            print('#########')
            print(f'Problem occurred with hdf5 file {hfile}')
            #raise Exception(f'Problem occurred with hdf5 file {hfile}')
        labels = labels + case_labels
        for caseID in caseIDs:
            symlinks[caseID] = hfile
    return symlinks, labels


def read_caseIDS_from_hfile(hfile):
    """Read a hdf5 file, returns 2 lists:
    - caseIDS: the name of the model (starting with BA_ or EL_)
    - label: the classification task target value (BIN_CLASS property)

    arguments:
    f5_file = path to the hdf5 file"""
    caseIDs = []
    labels = []
    with h5py.File(hfile, 'r') as h5:
        caseIDs = list(h5.keys())
        for case in caseIDs:
            try:
                labels.append(h5[case]["targets"]["BIN_CLASS"][()])
            except:
                print('#########')
                print(f'Problem occurred with case {case}')
                #raise Exception(f'Problem occurred with case {case}')
    return caseIDs, labels


def save_train_valid(train_idx, val_idx, test_idx, symlinks, path_out_folder,
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

    print("### Creating train.hdf5 file ###") 
    for i in train_idx:
        symlink_in_h5(i, symlinks[i], train_h5) 

    print("### Creating validation.hdf5 file ###") 
    for i in val_idx:
        symlink_in_h5(i, symlinks[i], val_h5) 

    print("### Creating test.hdf5 file ###") 
    for i in test_idx:
        symlink_in_h5(i, symlinks[i], test_h5) 

    train_h5.close()
    val_h5.close()
    test_h5.close()


def symlink_in_h5(idx, f1_path, f2):
    """Copy the selected keys (idx) of one hdf5 (f1) file into
     another hdf5 file (f2)
     idx = list of keys to copy
     f1 = handle of the first hdf5 file
     f2 = handle of the the second hdf5 file"""

    # Get the name of the parent for the group we want to copy
    f1 = h5py.File(f1_path, "r")
    group_path = f1[idx].name

    # Copy
    f2[group_path] = h5py.ExternalLink(f1_path, group_path)
    f1.close()


if __name__ == '__main__':
    # Combine the h5 files:
    symlinks, labels = h5_symlinks(a.features_path)
    n_splits=10

    #Make the output directories if they are not present already
    split_type = ('shuffled', 'clustered')[a.cluster]
    for split in range(0,n_splits):
        if not os.path.isdir(output_h5_path + f"/{split}"):
            os.makedirs(output_h5_path + f"/{split_type}/{split}")

    if a.cluster == False:
        all_cases = np.array(list(symlinks.keys()))
        labels = np.array(labels)
        indices = np.array(range(len(labels)))
        random.shuffle(indices)

        kfold = StratifiedKFold(n_splits = n_splits)
        i = 0
        for training_idx, test_idx in kfold.split(indices, labels[indices]):
            training_idx, validation_idx = train_test_split(training_idx, test_size=2/9, stratify=labels[training_idx])
            
            tr = all_cases[training_idx]
            va = all_cases[validation_idx]
            t = all_cases[test_idx]
            # create training and validation hdf5 files
            print(f"### SAVING SPLITS for {i} ###")
            save_train_valid(tr, va, t,symlinks, output_h5_path,
                f"shuffled/{i}/train.hdf5", 
                f"shuffled/{i}/valid.hdf5", 
                f"shuffled/{i}/test.hdf5") 
            i+=1
    else:
        df = pd.read_csv(f"{a.csv_file}")
        groups = [int(x) for x in set(df["cluster"])]
        all_cases = list(symlinks.keys())
        for i in groups:
            test_mask = (df["ID"].isin(all_cases)) & (df["cluster"] == i)
            test_cases = df[test_mask]["ID"]
            not_test_cases = np.array(df[test_mask == False]["ID"])
            random.shuffle(not_test_cases)
            ds_l = not_test_cases.shape[0]
            train_cases, validation_cases = np.split(
                not_test_cases, 
                [int(0.7*ds_l)]
            )
            print(f"### SAVING SPLITS FOR CLUSTER {i}")
            save_train_valid(train_cases, validation_cases, test_cases, symlinks, output_h5_path,
                f"clustered/{i}/train.hdf5",
                f"clustered/{i}/valid.hdf5",
                f"clustered/{i}/test.hdf5"
            )

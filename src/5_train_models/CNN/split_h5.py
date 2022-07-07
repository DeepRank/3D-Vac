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
    help="Path to the folder containing all h5 features generated by the 4/generate_features.py file.",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/"
)

arg_parser.add_argument("--cluster", "-c",
    help = "If provided, creates 10 train, validation, test splits for a leave one group out cross validation.",
    default = False,
    action = "store_true",
)

arg_parser.add_argument("--csv-file", "-d",
    help= "Name of db1. Needed only for clustered split.",
    default="BA_pMHCI.csv",
)
arg_parser.add_argument("--output-path", "-o",
    help="Destination path for the generated splits. For clustered will be in --output-path/clustered and in \
    --output-path/shuffled for shuffled. The clustered/0 to clustered/9 and shuffled/0 to shuffled/9 subfolders have \
    to be created before running this script. Default: \
    /projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/splits/"
)

a = arg_parser.parse_args()

# Change the combined_path and output_h5:
output_h5_path = a.output_path

__author__ = "Francesco Ambrosetti"
__email__ = "ambrosetti.francesco@gmail.com"

def h5_symlinks(input_path):
    """Combines HDF5 files from the input_path folder
    into a dictionaries of symlinks which then will be splited.
    """
    hfiles = glob.glob(f"{input_path}*.hdf5")
    symlinks = {}
    labels = []
    for hfile in hfiles:
        caseIDs, case_labels = read_caseIDS_from_hfile(hfile)
        labels = labels + case_labels
        for caseID in caseIDs:
            symlinks[caseID] = hfile
    return symlinks, labels


def read_caseIDS_from_hfile(hfile):
    """Read a hdf5 file and return the list of
    keys.
    f5_file = path to the hdf5 file"""
    caseIDs = []
    labels = []
    with h5py.File(hfile, 'r') as h5:
        caseIDs = list(h5.keys())
        for case in caseIDs:
            labels.append(h5[case]["targets"]["BIN_CLASS"][()])
    return caseIDs, labels


def save_train_valid(train_idx, val_idx, test_idx, symlinks, path_out_folder,
    train_f, valid_f, test_f
):
    """Save two subsets of the original hdf5 files for containing
    training and validation sets.
    the output file names are: train.hdf5, valid.hdf5

    train_idx = caseIDs for the training set (from split_train())
    val_idx = casedIDs for the validation set (from split_train())
    out_folder = path to the output folder
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
    if a.cluster == False:
        all_cases = np.array(list(symlinks.keys()))
        labels = np.array(labels)
        indices = np.array(range(len(labels)))
        random.shuffle(indices)

        kfold = StratifiedKFold(n_splits = 10)
        i = 0
        for training_idx, test_idx in kfold.split(indices, labels[indices]):
            training_idx, validation_idx = train_test_split(training_idx, test_size=2/9, stratify=labels[training_idx])
            
            tr = all_cases[training_idx]
            va = all_cases[validation_idx]
            t = all_cases[test_idx]
            # create training and validation hdf5 files
            print(f"### SAVING SPLITS for {i} ###")
            save_train_valid(tr, va, t,symlinks, output_h5_path,
            f"shuffled/{i}/train.hdf5", f"shuffled/{i}/valid.hdf5", f"shuffled/{i}/test.hdf5") 
            i+=1
    else:
        df = pd.read_csv(f"../../../data/external/processed/{a.csv_file}")
        groups = set(df["cluster"])
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
#!/usr/bin/env python
# Original author: Francesco Ambrosetti, modified by Daniil Lepikhov

import math
import h5py
import re
import random
import os
import glob

__author__ = "Francesco Ambrosetti"
__email__ = "ambrosetti.francesco@gmail.com"

def read_caseIDS_from_files(hfile):
    """Read a hdf5 file and return the list of
    keys with and without the rotation number (if present)
    f5_file = path to the hdf5 file"""
    caseIDs = []
    with h5py.File(hfile, 'r') as h5:
        caseIDs = list(h5.keys())
    return caseIDs


def get_proportions(peptides, train_part=0.80):
    """Get the training and validation keys by keeping
    the two sets balanced (same/similar bio and xtal interfaces)
    train_part = percentage of the dataset (bio_codes + xtal_codes
                 to be used for training (default is 0.80)"""

    # Get train and validation proportions
    total = len(peptides.shape[0])

    # Training set
    train_size = int(math.ceil(total * train_part))

    # Extract cases keeping valid and train balanced
    train_cases = random.sample(peptides, train_size)

    # Get validation cases
    valid_cases = [x for x in peptides if x not in train_cases]

    # Print everything
    print(f'Training set cases: {len(train_cases)}')
    print(f'Validation set cases: {len(valid_cases)}')

    return train_cases, valid_cases


def split_train(peptides, generated_features_path, train_part=0.80):
    """Get the training and validation keys by keeping
    the two sets balanced (same/similar bio and xtal interfaces)
    bio_codes = list of codes of the bio interfaces
    xtal_codes = list of codes of the xtal interfaces
    train_part = percentage of the dataset (bio_codes + xtal_codes
                 to be used for training (default is 0.80)"""

    # Read files
    all_cases = read_caseIDS_from_files(generated_features_path)

    print(f'Loaded pMHCI complexes: {len(all_cases)}')

    # Get train and validation IDs (without rotation)
    train_ids, valid_ids = get_proportions(peptides, train_part)

    # Sort IDs
    train_ids.sort()
    valid_ids.sort()

    print(f'Training set: {len(train_ids)}')
    print(f'Validation set: {len(valid_ids)}')

    return train_ids, valid_ids


def save_train_valid(train_idx, val_idx, out_folder):
    """Save two subsets of the original hdf5 files for containing
    training and validation sets.
    the output file names are: train.hdf5, valid.hdf5

    bio_h5 = original hdf5 file for the bio interfaces
    xtal_h5 = original hdf5 file for the xtal interfaces
    train_idx = caseIDs for the training set (from split_train())
    val_idx = casedIDs for the validation set (from split_train())
    out_folder = path to the output folder"""

    # Create new hd5f files for the bio interfaces
    train_h5 = h5py.File(os.path.join(out_folder, 'train.hdf5'), 'w')
    val_h5 = h5py.File(os.path.join(out_folder, 'valid.hdf5'), 'w')
    with h5py.File(bioh5, 'r') as f1:
        print('#### Creating Training  set file for bio ####')
        subset_h5(bio_train, f1, th5_bio)

        print('#### Creating Validation  set file for bio ####')
        subset_h5(bio_valid, f1, val5_bio)
    th5_bio.close()
    val5_bio.close()


def subset_h5(idx, f1, f2):
    """Copy the selected keys (idx) of one hdf5 (f1) file into
     another hdf5 file (f2)
     idx = list of keys to copy
     f1 = handle of the first hdf5 file
     f2 = handle of the the second hdf5 file"""

    for b in idx:
        print(f'copying: {b}')

        # Get the name of the parent for the group we want to copy
        group_path = f1[b].parent.name

        # Check that this group exists in the destination file; if it doesn't, create it
        # This will create the parents too, if they don't exist
        group_id = f2.require_group(group_path)

        # Copy
        f1.copy(b, group_id)


if __name__ == '__main__':

    # Define input hdf5 files
    train_bio = '/projects/0/deepface/MANY/hdf5/manybio_au30.hdf5'
    train_xtal = '/projects/0/deepface/MANY/hdf5/manyxtal_au30.hdf5'

    # Get training and validation keys
    tr, va = split_train(train_bio, train_xtal, 0.80)

    # Save keys into files
    with open('train_caseIDS.txt', 'w') as x:
        for element in tr:
            x.write(element)
            x.write('\n')
    with open('valid_caseIDS.txt', 'w') as y:
        for ele in va:
            y.write(ele)
            y.write('\n')

    # Create training and validation hdf5 files
    save_train_valid(train_bio, train_xtal, tr, va, out_folder='/projects/0/deepface/MANY/hdf5/')

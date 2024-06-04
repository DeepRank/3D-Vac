import os
import glob
import h5py
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split


csvs_path = '/projects/0/einf2380/data/external/processed/I/CrossValidations'
parse_hdf5=False

#%%
#CNN
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

ids_csv = '/projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative_only_eq.csv'
df = pd.read_csv(ids_csv)
df['label'] = np.where(df['measurement_value'] < 500, 1.0, 0.0)
full_dataset = df[['ID', 'label']]
all_ids = df.ID.tolist()

if parse_hdf5:
    hfiles = glob.glob(os.path.join('/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/exp_nmers_all_HLA_quantitative', '*.hdf5'))
    IDs, labels = [], []
    for hfile in hfiles:
        print(hfile)
        caseIDs, case_labels = read_caseIDS_from_hfile(hfile, all_ids)
        IDs.extend(caseIDs)
        labels.extend(case_labels)

    dataset = full_dataset[full_dataset['ID'].isin(IDs)]
    dataset.to_csv(f'{csvs_path}/full_dataset.csv', index=False)

else:
    dataset = pd.read_csv(f'{csvs_path}/full_dataset.csv')
#%% Make the shuffled folds

X = dataset['ID']
Y = dataset['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    stratify=Y, 
                                                    test_size=0.10,
                                                    random_state=42)

train_df = pd.DataFrame({'ID': X_train, 'label':Y_train})
test_df = pd.DataFrame({'ID': X_test, 'label':Y_test})
test_df.to_csv(f'{csvs_path}/Shuffled/test.csv', index=False)

#seeds = [3, 18, 31, 42, 65]
for fold in range(1,6):
    try:
        os.mkdir(f'{csvs_path}/Shuffled/{fold}')
    except Exception as e:
        print(e)
        
    fold_X_train, fold_X_val, fold_Y_train, fold_Y_val = train_test_split(train_df['ID'], train_df['label'],
                                                    stratify=train_df['label'], 
                                                    test_size=0.15,
                                                    random_state=fold)
    
    fold_train_df = pd.DataFrame({'ID': fold_X_train, 'label':fold_Y_train})
    fold_train_df.to_csv(f'{csvs_path}/Shuffled/{fold}/train.csv', index=False)
    
    fold_val_df = pd.DataFrame({'ID': fold_X_val, 'label':fold_Y_val})
    fold_val_df.to_csv(f'{csvs_path}/Shuffled/{fold}/validation.csv', index=False)
    
#%% Make the AlleleClustered folds

AlleleClustered_test = pd.read_csv('/projects/0/einf2380/data/external/processed/I/experiments/BA_pMHCI_human_quantitative_only_eq_pseudoseq_clustered_test.csv')

test_df = dataset[dataset['ID'].isin(AlleleClustered_test['ID'])]
test_df.to_csv(f'{csvs_path}/AlleleClustered/test.csv', index=False)

train_df = dataset[~dataset['ID'].isin(AlleleClustered_test['ID'])]

for fold in range(1,6):
    try:
        os.mkdir(f'{csvs_path}/AlleleClustered/{fold}')
    except Exception as e:
        print(e)
        
    fold_X_train, fold_X_val, fold_Y_train, fold_Y_val = train_test_split(train_df['ID'], train_df['label'],
                                                    stratify=train_df['label'], 
                                                    test_size=0.15,
                                                    random_state=fold)
    
    fold_train_df = pd.DataFrame({'ID': fold_X_train, 'label':fold_Y_train})
    fold_train_df.to_csv(f'{csvs_path}/AlleleClustered/{fold}/train.csv', index=False)
    
    fold_val_df = pd.DataFrame({'ID': fold_X_val, 'label':fold_Y_val})
    fold_val_df.to_csv(f'{csvs_path}/AlleleClustered/{fold}/validation.csv', index=False)

#%% Check

def check_overlapping_ids(df1, df2):
    set1 = set(df1['ID'])
    set2 = set(df2['ID'])

    # Find the intersection of the two sets
    intersection = set1 & set2

    # Count the number of overlapping IDs
    num_overlapping_ids = len(intersection)

    print(num_overlapping_ids)
    return num_overlapping_ids

def check_pos_neg_ratio(df):
    pos = df['label'].sum()
    neg = (df['label'] == 0.0).sum()

    # Calculate the pos/neg ratio
    if neg == 0:
        pos_neg_ratio = float('inf')  # Handle division by zero if there are no negative labels
    else:
        pos_neg_ratio = pos / neg

    print(f"Number of positive labels: {pos}")
    print(f"Number of negative labels: {neg}")
    print(f"Positive/Negative Ratio: {pos_neg_ratio}")

path = '/projects/0/einf2380/data/external/processed/I/CrossValidations/AlleleClustered/%i/%s.csv'
check_overlapping_ids(pd.read_csv(path %(1, 'validation')), pd.read_csv(path %(3, 'validation')))
check_overlapping_ids(pd.read_csv(path %(1, 'validation')), pd.read_csv(path %(1, 'train')))

check_pos_neg_ratio(pd.read_csv(path %(1, 'validation')))
check_pos_neg_ratio(pd.read_csv(path %(1, 'train')))


import torch
import argparse
from torch import nn
import copy
import os.path as path
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.I.classification.seq import data_path # path to the data folder relative to the location of the __init__.py file
# import multiprocessing as mp
from mpi4py import MPI
from deeprank.learn import Dataset, NeuralNet
import h5py
import pandas as pd

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description="Fully connected layer to generate a model which predicts binders based on atomic features \
    from HDF5 file. Takes as input the path to the hdf5 train, validation and test. \
    Uses the --csv-file as the input to define binders or not.\
    The default threshold for binders is 500."
)

arg_parser.add_argument("--csv-file", "-f",
    help="Absolute path of the csv file for target calculation.",
    default=f"BA_pMHCI.csv"
)
arg_parser.add_argument("--splits-path", "-s",
    help="Path to train.hdf5, validation.hdf5 and test.hdf5 files. Default \
    /projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/splits",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/splits"
)
arg_parser.add_argument("--threshold", "-t",
    help="Binding affinity threshold to define binders, default 500.",
    type=float,
    default=500
)
arg_parser.add_argument("--cluster", "-c",
    help="By providing this argument, will perform a scikit LeavOneGroupOut crossvalidation grouped by cluster, shuffled KFold otherwise.",
    default=False,
    action="store_true"
)
arg_parser.add_argument("--neurons", "-N",
    help="Number of neurons per layer. Default 1000.",
    default=5,
    type=int
)
arg_parser.add_argument("--batch", "-B",
    help="Batch size. Default 64.",
    type=int,
    default=64
)
arg_parser.add_argument("--epochs", "-E",
    help="Number of times to iterate through the whole dataset. Default 150.",
    default=10,
    type=int
)
arg_parser.add_argument("--model-name", "-o",
    help="Name of the model name.",
    required=True
)

a = arg_parser.parse_args()

# MPI INITIALIZATION
#-------------------

mpi_conn = MPI.COMM_WORLD
rank = mpi_conn.Get_rank()
size = mpi_conn.Get_size()
datasets = []

best_model = {
    "validation_rate": 0,
    "model": None,
    "best_epoch": None,
    "test_data": None
}

# onehot (sparse) encoding

# FUNCTIONS AND USEFUL STUFF
#----------------------------

# hyperparamaters (all defined as arguments)
neurons_per_layer = a.neurons
batch = a.batch
epochs = a.epochs

# DATA PREPROCESSING
#----------------------------------------------
if rank == 0:
    print("Loading data...")
    csv_path = path.abspath(f"{data_path}external/processed/{a.csv_file}")
    csv_df = pd.read_csv(csv_path)
    splits = {
        "train" : {
            "h5": h5py.File(f"{a.splits_path}/train.hdf5", "r"),
            "labels": []
        },
        "val" : {
            "h5": h5py.File(f"{a.splits_path}/valid.hdf5", "r"),
            "labels": []
        },
        "test" : {
            "h5": h5py.File(f"{a.splits_path}/test.hdf5", "r"),
            "labels": []
        }
    }
    for split in splits:
        cases = list(splits[split]["h5"].keys())
        split_df = csv_df["ID"].isin(cases) 
        print(len(cases))
        print(len(split_df))
        # splits[split]["labels"] = [(0.,1.)[value < a.threshold] for value in split_df["measurement_value"]]
    print("Data loaded, splitting into unique test datasets...")

    # SEPARATE TRAIN VALIDATION AND TEST DATASETS
    # -------------------------------------------
    if a.cluster == False:
        print("generating dataset")
    else:
        print("generating dataset clustered")

# CREATE MULTIPROCESSING
#-----------------------
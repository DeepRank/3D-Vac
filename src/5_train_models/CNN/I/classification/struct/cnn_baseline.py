import torch
import argparse
from torch import nn
import copy
import os.path as path
import os
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.I.classification.seq import data_path # path to the data folder relative to the location of the __init__.py file
from CNN.models import CnnClassificationBaseline
# import multiprocessing as mp
from mpi4py import MPI
from deeprank.learn import DataSet, NeuralNet
import pandas as pd
from deeprank.learn.modelGenerator import *
from deeprank.learn.model3d import cnn_class

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description="Fully connected layer to generate a model which predicts binders based on atomic features \
    from HDF5 file. Takes as input the path to the hdf5 train, validation and test. \
    Uses the --csv-file as the input to define binders or not.\
    The default threshold for binders is 500."
)

arg_parser.add_argument("--splits-path", "-s",
    help="Path to train.hdf5, validation.hdf5 and test.hdf5 files. Default \
    /projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/splits",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/splits"
)
arg_parser.add_argument("--cluster", "-c",
    help="By providing this argument, will perform a scikit LeavOneGroupOut crossvalidation grouped by cluster, shuffled KFold otherwise.",
    default=False,
    action="store_true"
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
arg_parser.add_argument("--output-dir", "-o",
    help="Name of the folder where 10 subfolders will be created for each cross validation.",
    required = True,
)

a = arg_parser.parse_args()

# MPI INITIALIZATION
#-------------------
mpi_conn = MPI.COMM_WORLD
rank = mpi_conn.Get_rank()
size = mpi_conn.Get_size()
datasets = []

# LOAD DATA
#----------
train_db = (f"{a.splits_path}/shuffled/{rank}/train.hdf5", f"{a.splits_path}/clustered/{rank}/train.hdf5")[a.cluster]
val_db = (f"{a.splits_path}/shuffled/{rank}/valid.hdf5", f"{a.splits_path}/clustered/{rank}/valid.hdf5")[a.cluster]
test_db = (f"{a.splits_path}/shuffled/{rank}/test.hdf5", f"{a.splits_path}/clustered/{rank}/test.hdf5")[a.cluster]

# create dirs:
if rank == 0:
    for i in range(10):
        folder = f"{a.output_dir}/{i}"
        try:
            os.makedirs(folder)
        except OSError as error:
            print(f"folder {folder} already created")
outdir = f"{a.output_dir}/{rank}"

data_set = DataSet(train_database=train_db,
    test_database = test_db,
    valid_database = val_db,
    chain1="M",
    chain2="P",
    grid_info=(35,30,30),
    select_feature="all",
    select_target="BIN_CLASS",
    normalize_features = True,
    normalize_targets = False,
    pair_chain_feature = None,
    mapfly = False,
    tqdm= True,
    clip_features=False,
    process=True,
)

model = NeuralNet(data_set=data_set,
    model= CnnClassificationBaseline,
    task="class",
    chain1="M",
    chain2="P",
    cuda=True,
    ngpu=1,
    plot=True,
    save_classmetrics=True,
    outdir=outdir
)

model.train(
    nepoch = 10,
    divide_trainset=None,
    train_batch_size=32,
    save_model="best",
    save_epoch="all",
    hdf5="metrics.hdf5",
)
# START TRAINING
#---------------

conv_layers = []
conv_layers.append()
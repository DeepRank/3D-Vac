import argparse
import os.path as path
import os
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.I.classification.seq import data_path # path to the data folder relative to the location of the __init__.py file
from CNN.models import CnnClassificationBaseline
# import multiprocessing as mp
from mpi4py import MPI
from deeprank.learn import DataSet, NeuralNet
from deeprank.learn.modelGenerator import *

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description = "Script used to load the train, validation and test HDF5 files generated with 5/CNN/split_train_h5.py \
    and train 10 models using CnnClassificationBaseline architecture from CNN/models.py file. \
    The script should be run using cnn_baseline_cuda.sh or cnn_baseline.sh. \
    The script expects both shuffled and clustered HDF5 splits to be in the {--split-path}/shuffled/<0:9>/ \
    and in the {--split-path}/clustered/<0:9>/ folders (works only for 10 fold xvalidation). By default this script \
    will perform a shuffled cross validation, for a clustered please provide --cluster argument.",
)

arg_parser.add_argument("--splits-path", "-s",
    help="Path to shuffled and clustered folders containing subfolders named from 0 to 9 each with \
    train.hdf5, valid.hdf5 and test.hdf5 files as the splits for each fold. Default path \
    /projects/0/einf2380/data/pMHCI/features_output_folder/hla_a_02_01_9_length_peptide/splits",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/splits"
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
    help="Number of times to iterate through the whole dataset. Default 10.",
    default=10,
    type=int
)
arg_parser.add_argument("--output-dir", "-o",
    help="Name of the folder where 10 subfolders will be created for each cross validation. Required.",
    required = True,
)
arg_parser.add_argument("--with-cuda", "-C",
    help="By default True. It is not provided directly by the user but is hardcoded in cnn_baseline.sh or cnn_baseline.cuda.sh\
    To use cuda cores please run the cnn_baseline_cuda.sh (--with-cuda automatically set to True), otherwise \
    cnn_baseline.sh file (--with-cuda set to False).",
    default=False,
    action="store_true"
)

a = arg_parser.parse_args()

# MPI INITIALIZATION
#-------------------
mpi_conn = MPI.COMM_WORLD
rank = mpi_conn.Get_rank()
size = mpi_conn.Get_size()
datasets = []

# handle printing messages only for one task so the logs are not messy:
if rank != 0:
    sys.stdout = open(os.devnull, 'w')

# LOAD DATA
#----------

# if the --cluster argument is provided, loads train, valid and test from the --splits-path/cluster/ folder
train_db = (f"{a.splits_path}/shuffled/{rank}/train.hdf5", f"{a.splits_path}/clustered/{rank}/train.hdf5")[a.cluster]
val_db = (f"{a.splits_path}/shuffled/{rank}/valid.hdf5", f"{a.splits_path}/clustered/{rank}/valid.hdf5")[a.cluster]
test_db = (f"{a.splits_path}/shuffled/{rank}/test.hdf5", f"{a.splits_path}/clustered/{rank}/test.hdf5")[a.cluster]

# create dirs:
if rank == 0:
    for i in range(10):
        folder = f"./trained_models/{a.output_dir}/{i}"
        try:
            os.makedirs(folder)
        except OSError as error:
            print(f"folder {folder} already created")
outdir = f"./trained_models/{a.output_dir}/{rank}"

data_set = DataSet(train_database=train_db,
    test_database = test_db,
    valid_database = val_db,
    chain1 = "M",
    chain2 = "P",
    grid_info = (35,30,30),
    select_feature = "all",
    select_target = "BIN_CLASS",
    normalize_features = True,
    normalize_targets = False,
    pair_chain_feature = None,
    mapfly = False,
    tqdm = True,
    clip_features = False,
    process = True,
)

model = NeuralNet(data_set=data_set,
    model = CnnClassificationBaseline,
    task = "class",
    chain1 = "M",
    chain2 = "P",
    cuda = bool(a.with_cuda),
    ngpu = (0,1)[a.with_cuda],
    plot = True,
    save_classmetrics = True,
    outdir = outdir
)

model.train(
    nepoch = 50,
    divide_trainset = None,
    train_batch_size = 32,
    save_model = "best",
    save_epoch = "all",
    hdf5 = "metrics.hdf5",
)
# START TRAINING
#---------------
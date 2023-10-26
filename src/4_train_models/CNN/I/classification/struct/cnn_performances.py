import argparse
import os.path as path
import os
import sys
sys.path.append(path.abspath("../../../../"))
from CNN.CNN_models import *
# import multiprocessing as mp
# from mpi4py import MPI
from CNN.NeuralNet import NeuralNet
from deeprank.learn.modelGenerator import *

# DEFINE CLI ARGUMENTS
#---------------------

arg_parser = argparse.ArgumentParser(
    description = "This script is used to generate the results of the trained model using cnn_baseline.py.",
)

arg_parser.add_argument("--pretrained-model", "-m",
    help="Path to pretrained model to test",
    required=True
)
arg_parser.add_argument("--test-hdf5", "-t",
    help="Path to test hdf5",
    required=True
)
arg_parser.add_argument("--architecture", "-a",
    help="Model architecture (NeuralNet class) to use",
    required=True
)
# arg_parser.add_argument("--splits-path", "-s",
#     help="Path to shuffled and clustered folders containing subfolders named from 0 to 9 each with \
#     train.hdf5, valid.hdf5 and test.hdf5 files as the splits for each fold. Default path \
#     /projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/splits",
#     default="/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide/splits"
# )
# arg_parser.add_argument("--cluster", "-c",
#     help="By providing this argument, will perform a scikit LeavOneGroupOut crossvalidation grouped by cluster, shuffled KFold otherwise.",
#     default=False,
#     action="store_true"
# )
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
# mpi_conn = MPI.COMM_WORLD
# rank = mpi_conn.Get_rank()
# size = mpi_conn.Get_size()
#datasets = []

# handle printing messages only for one task so the logs are not messy:
#if rank != 0:
#sys.stdout = open(os.devnull, 'w')

# LOAD DATA
#----------

# if the --cluster argument is provided, loads train, valid and test from the --splits-path/cluster/ folder
#test_db = (f"{a.splits_path}/shuffled/{rank}/test.hdf5", f"{a.splits_path}/clustered/{rank}/test.hdf5")[a.cluster]

# create dirs:
# if rank == 0:
#     for i in range(10):
#         folder = f"./tested_models/{a.output_dir}/{i}"
#         try:
#             os.makedirs(folder)
#         except OSError as error:
#             print(f"folder {folder} already created")

outdir = f"./tested_models/{a.output_dir}/"
try:
    os.makedirs(outdir)
except FileExistsError:
    pass

model = NeuralNet(a.test_hdf5,
    model = a.architecture,
    cuda = bool(a.with_cuda),
    ngpu = (0,1)[a.with_cuda],
    outdir = outdir,
    pretrained_model=a.pretrained_model,
)

model.test()
# else:
#     print(f"Skipping fold {rank}")
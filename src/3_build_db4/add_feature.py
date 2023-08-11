from deeprank.generate import DataGenerator
#from sequence_feature import SequenceFeature
from glob import glob
import argparse
#from mpi4py import MPI
#from math import ceil
import numpy as np
from joblib import Parallel, delayed

arg_parser = argparse.ArgumentParser(
    description = "Adds a given feature to deeprank hdf5 files",
)

arg_parser.add_argument("--features-name", "-f",
    help="Feature name to add.",
    required=True,
    nargs='+',
)

arg_parser.add_argument("--data-path", "-d",
    help="Path to folders containing the main hdf5 files (NOT the symlinked ones) which need the feature to be added",
    required=True,
)

arg_parser.add_argument("--n-cores", "-n",
    help="Number of cores",
    default=1,
    type=int,
)

def add_feature(features_name, h5file):
    database = DataGenerator(compute_features=args.features_name,
                            hdf5=h5file, chain1='M', chain2='P')

    database.add_feature()
    database.map_features()
    
# mpi_conn = MPI.COMM_WORLD
# rank = mpi_conn.Get_rank()
# size = mpi_conn.Get_size()

args = arg_parser.parse_args()

hdf5_files = glob(args.data_path + '/*.hdf5')

Parallel(n_jobs = args.n_cores, verbose = 1)(delayed(add_feature)(args.features_name, h5file) for h5file in hdf5_files)
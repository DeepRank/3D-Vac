import argparse
import glob
import h5py
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import os
from mpi4py import MPI

conn = MPI.COMM_WORLD
size = conn.Get_size()
rank = conn.Get_rank()

arg_parser = argparse.ArgumentParser(description="""
    Plots the distribution of max values or values of each feature in report/figures/h5explorer/{name_of_exploration}
    for a given set of h5 files.
""")
arg_parser.add_argument("--path-to-h5", "-p",
    help="Path to where h5 features are stored. Without closing `/`.",
    default="/projects/0/einf2380/data/pMHCI/features_output_folder/CNN/hla_a_02_01_9_length_peptide"
)
arg_parser.add_argument("--name", "-n",
    help="Name of the subfolder in report/figures/h5explorer/ to save the plots.",
    default="test"
)
arg_parser.add_argument("--plot-max", "-M",
    help="Provide this argument to plot only the max value for each feature. Otherwise plots everything.",
    default=False,
    action="store_true",
)
arg_parser.add_argument("--use-pickle", "-s",
    help="If provided, doesn't generate the key value dictionnary containing all features values.",
    default=False,
    action="store_true"
)

a = arg_parser.parse_args()

# Retrieve all hdf5 files with a simple command:
if rank == 0:
    print(f"Size: {size}")
    h5_files = np.array(glob.glob(f"{a.path_to_h5}/*.hdf5"))
    h5_files_bulk = np.array_split(h5_files, size)
else:
    h5_files_bulk = None

h5_files_bulk = conn.scatter(h5_files_bulk, root=0)

feature_list = []
feature_values = []

#populating the k_mv_d or k_v_d with values from each hdf5 file:
print(f"Populating features_values on {rank}")
for i,f in enumerate(h5_files_bulk):
    h5 = h5py.File(f)
    for m in h5.keys(): # iterating through cases
        if i == 0:
            features = list(h5[f"{m}/mapped_features/Feature_ind"].keys())
            feature_list = [[]]*len(features)
            feature_values = [[]]*len(features)
        if i < 5:
            for k,f in enumerate(features): # iterating through features
                values = h5[f"{m}/mapped_features/Feature_ind/{f}/value"][()].flatten().tolist()
                if a.plot_max:
                    if values.shape[0] == 0:
                        feature_values[k].append(0)
                    else:
                        feature_values[k].append(values.max())
                else:
                    feature_values[k].extend(values)
print(f"Finished populating features_values on {rank}")
# make the dir if not already existing:
plot_dir = f"../../reports/figures/h5explorer/{a.name}"
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

#Gather data from workers:
all_values = conn.gather(feature_values)
if rank == 0:
    print(f"type of all_values: {type(all_values)}, len: {len(all_values)}")
    all_features = [[]]*len(features)
    # pkl.dump(all_values, open("./all_feats.pkl", "wb"))
    for feature_groups in all_values:
        for i,values in enumerate(feature_groups):
            all_features[i].extend(values) 
    print("Creating all_features np array..")
    all_features = np.array(all_features)
    print("Array created. Splitting into workers for ploting..")
    features_bulk = np.array_split(all_features, size)
else:
    features_bulk = None

features_bulk = conn.scatter(features_bulk, root=0)
for f in features_bulk:
    values_array = np.array(f)
    print(f"values_array shape on {rank}: {values_array.shape}")

# if a.plot_max:
#     # plot each feature:
#     for feature in feature_list():
#         values = np.array(k_mv_d[feature])
#         plt.hist(values, bins=100, log=True)
#         plt.title(f"Distribution of max {feature} values, mapped on the grid")
#         plt.xlabel("Max values for each model")
#         plt.ylabel("Log of bin count")

#         # save the plot:
#         # plt.savefig(f"{plot_dir}/{feature}.png")
#         plt.savefig("./test.png")
#         plt.close()

# else:
#     for i,f in enumerate(k_v_d.keys()):
#         # values = np.array(k_v_d[f])
#         print(f"Feature: {f}, size: {values.shape}, min value: {np.amin(values)}, max value: {np.amax(values)}")

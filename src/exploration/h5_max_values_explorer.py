import argparse
import glob
import h5py
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

arg_parser = argparse.ArgumentParser(description="""
    Plots the distribution of each feature in report/figures/h5explorer/{name_of_exploration}
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
arg_parser.add_argument("--plot", "-P",
    help="Provide this argument to save plots in the --name subfolder. By default it won't plot.",
    default="False",
    action="store_true"
)

a = arg_parser.parse_args()

# Retrieve all hdf5 files with a simple command:
h5_files = glob.glob(f"{a.path_to_h5}/*.hdf5")

# k_mv_d = {} # key_max_value_dictionary
k_mv_d = {} # key_max_value_dictionary
#populating the k_mv_d with values from each hdf5 file:
for i,f in enumerate(h5_files):
        h5 = h5py.File(f)
        for m in h5.keys():
            if i == 0:
                keys = list(h5[f"{m}/mapped_features/Feature_ind"].keys())
                k_mv_d = {k:[] for k in keys}
            for k in keys:
                values = h5[f"{m}/mapped_features/Feature_ind/{k}/value"][()]
                if "vdwaals" in k:
                    max_values = values[values > 8000]
                    if max_values.shape[0] >= 1:
                        print(f"The model {m} has {max_values.shape[0]} values superior to 8000.")
                if values.shape[0] == 0:
                    k_mv_d[k].append(0)
                else:
                    k_mv_d[k].append(values.max())

if a.plot:
    # make the dir if not already existing:
    plot_dir = f"../../reports/figures/h5explorer/{a.name}"
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    # plot each feature:
    for feature in k_mv_d.keys():
        values = np.array(k_mv_d[feature])
        plt.hist(values, bins=100, log=True)
        plt.title(f"Distribution of max {feature} values, mapped on the grid")
        plt.xlabel("Max values for each model")
        plt.ylabel("Log of bin count")

        # save the plot:
        plt.savefig(f"{plot_dir}/{feature}.png")
        plt.close()
import argparse
import glob
import h5py

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

a = arg_parser.parse_args()

# Retrieve all hdf5 files with a simple command:
h5_files = glob.glob(f"{a.path_to_h5}/*.hdf5")
print(len(h5_files))

features = []
values = []
for i,f in enumerate(h5_files):
    if i ==0:
        h5 = h5py.File(f)
        for m in h5.keys():
            features.append(h5[m].keys())
print(features)
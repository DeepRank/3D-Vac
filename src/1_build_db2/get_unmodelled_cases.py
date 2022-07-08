import os
import sys
import glob
import argparse
import pandas as pd

arg_parser = argparse.ArgumentParser(description="Script used to generate a list of unmodelled p:MHC complexes looking at \
the output folder and the initial db1.")

arg_parser.add_argument("--csv-file", "-f",
    help="Path to db1 containing the p:MHC complexes to model.",
    default="BA_pMHCI.csv",
)

a = arg_parser.parse_args()

#1. Open cases file
df = pd.read_csv(f"../../data/external/processed/{a.csv_file}")
all_cases = len(df)

#2. Open output folder. Check how many cases have model number 20. 
already_modelled = []
model_dirs = glob.glob('/projects/0/einf2380/data/pMHCI/models/BA/*/*');
for folder in model_dirs:
    n_structures = len(glob.glob(f"{folder}/*.BL*.pdb"))
    if n_structures == 20:
        case = "_".join(folder.split("/")[-1].split("_")[0:2])
        df.drop(df[df["ID"]==case].index, inplace=True)

print(f"Initial number of cases: {all_cases}")
print(f'Unmodelled: {len(df)}')

# #3. Write new input file without cases already modelled.
df.to_csv("../../data/external/processed/to_model.csv", index=False)
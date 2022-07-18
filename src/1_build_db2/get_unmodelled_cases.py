import os
import sys
import glob
import argparse
import pandas as pd

arg_parser = argparse.ArgumentParser(description="Script used to generate a list of unmodelled p:MHC complexes by comparing \
the output folder and the initial db1. A case is considered modelled if 19 pdb structures were generated. Here 19 is the \
threshold because modeller sometimes is able to generate only 19/20 in very extreme cases (6 out of 11K cases). Therefore, \
even extreme cases are considered modelled.")

arg_parser.add_argument("--csv-file", "-f",
    help="Path to db1 containing the p:MHC complexes to model.",
    default="BA_pMHCI.csv",
)

a = arg_parser.parse_args()

#1. Open cases file
df = pd.read_csv(f"../../data/external/processed/{a.csv_file}")
all_cases = len(df)

#2. Open output folder. If the folder doesn't have 20 pdb files, it is considered as unmodelled. 
already_modelled = []
model_dirs = glob.glob('/projects/0/einf2380/data/pMHCI/models/BA/*/*');
for folder in model_dirs:
    n_structures = len(glob.glob(f"{folder}/*.BL*.pdb"))
    if n_structures >= 19 and n_structures <= 20: # the n_structures <= 20 is to be sure that no more than 20 structures are
        # generated
        case = "_".join(folder.split("/")[-1].split("_")[0:2])
        df.drop(df[df["ID"]==case].index, inplace=True) # the initial list of cases is reduced with the case.

print(f"Initial number of cases: {all_cases}")
print(f'Unmodelled: {len(df)}')

# #3. Write new input file without cases already modelled.
df.to_csv("../../data/external/processed/to_model.csv", index=False) # the initial list of cases without the modelled cases
# is returned
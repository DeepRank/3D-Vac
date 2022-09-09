import os
import sys
import glob
import argparse
import pandas as pd

arg_parser = argparse.ArgumentParser(description="Script used to generate a list of unmodelled p:MHC complexes by comparing \
the output folder and the initial db1. A case is considered modelled if 19 pdb structures were generated. Here 19 is the \
threshold because modeller sometimes is able to generate only 19/20 in very extreme cases (6 out of 11K cases). Therefore, \
even extreme cases are considered modelled. The logs on how much cases are still to model is available at the printed logs \
which is at project_folder/data/modelling_logs/clean_models_job.log")

arg_parser.add_argument("--csv-file", "-f",
    help="Path to db1 containing the p:MHC complexes to model.",
    default="../../data/external/processed/I/BA_pMHCI.csv",
)
arg_parser.add_argument("--to-model", "-t",
    help="Path to to_model.csv file ",
    default="../../data/external/processed/I/to_model.csv",
)
arg_parser.add_argument("--update-csv", "-u",
    help="If this argument is provided, the `to_model.csv` file is updated with the unmodelled cases and the number \
    of cases to model is printed into the log file. If not provided, it prints the number of cases to model without \
    updating the `to_model.csv`",
    default=False,
    action="store_true"
)
arg_parser.add_argument("--models-dir", "-m",
    help="Path to the BA or EL folder where the models are generated",
    default="/projects/0/einf2380/data/pMHCI/models/BA",
)

a = arg_parser.parse_args()

#1. Open cases file
df = pd.read_csv(f"{a.csv_file}")
all_cases = len(df)

#2. Open output folder. If the folder doesn't have 20 pdb files, it is considered as unmodelled. 
already_modelled = []
model_dirs = glob.glob(os.path.join(a.models_dir, '/*/*'))
for folder in model_dirs:
    n_structures = len(glob.glob(f"{folder}/*.BL*.pdb"))
    if n_structures >= 19 and n_structures <= 20: # the n_structures <= 20 is to be sure that no more than 20 structures are
        # generated
        case = "_".join(folder.split("/")[-1].split("_")[0:2])
        df.drop(df[df["ID"]==case].index, inplace=True) # the initial list of cases is reduced with the case.

print(f"Initial number of cases: {all_cases}")
print(f'Unmodelled: {len(df)}')

# #3. Write new input file without cases already modelled.
if a.update_csv:
    df.to_csv(a.to_model, index=False) # the initial list of cases without the modelled cases
# is returned
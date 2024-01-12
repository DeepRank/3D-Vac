import argparse
import pandas as pd
from db1_to_db2_path import assign_outfolder

# list of arguments:
arg_parser = argparse.ArgumentParser(
    description='Step 1 of db1. Generate a csv file with an ID column. This is done seperately for each measurement (BA/MS). \
    The source csv is a file containing at least the following columns: [MHC-allele,peptide,measurement-value] \
    The resulting csv serves as input for db1 step 2')
arg_parser.add_argument(
    "--source-csv", "-f",
    help="Name of the MHCflurry dataset csv file in data/external/unprocessed if different from the default.",
    default="../../data/external/unprocessed/curated_training_data.no_additional_ms.csv"
)
arg_parser.add_argument(
    "--output-csv", "-d",
    help="Name of destination csv with filtered entries. If provided, will be saved. \
    Otherwise this script can be used to visualize the applied filters.",
    default = None
)
arg_parser.add_argument("--measurement_type", "-t",
    help='The type of measurement to filter from the input csv. Choices: "BA" or "MS"',
    choices=["MS", "BA"],
    required=True
)
arg_parser.add_argument("--prefix", "-i",
    help="The prefix for the ID in the output csv",
    choices=["MS", "BA"],
    required=True
)

a = arg_parser.parse_args()

input_csv_df = pd.read_csv(f"{a.source_csv}")
ids = list(range(len(input_csv_df)))
input_csv_df.insert(0, column="ID", value=[f"{a.prefix}-{id+1}" for id in ids]) # give an ID to each entry
# PANDORA generated models location (provided as an argument for the modeling, among peptide and MHC allele):
input_csv_df["db2_folder"] = [f"/projects/0/einf2380/data/pMHCI/models/{a.prefix}/{assign_outfolder(id+1)}" for id in ids]

# filter based on user input
query = []
if a.measurement_type == "BA":
    query.append("measurement_kind == 'affinity'")
else:
    query.append("measurement_kind == 'mass_spec'")

# construct the complete query
query_string = "&".join(query)

print(f"filtering the following: {query_string}")
# filter the dataframe
input_csv_df = input_csv_df.query(query_string)

# apply the allele and length of peptide filter:
output_csv_df = input_csv_df

#save the csv:
if a.output_csv:
    output_csv_df.to_csv(f"{a.output_csv}", index=False)
    print(f"file {a.output_csv} with {len(output_csv_df)} entries saved in data/external/processed/")
else:
    print(output_csv_df)
import argparse
import pandas as pd
from db1_to_db2_path import assign_outfolder

# list of arguments:
arg_parser = argparse.ArgumentParser(
    description='Build a p:MHC Binding Affinity csv file for BA only with a precise value ("=" inequallity) from MHCFlurry dataset. For NetMHCpan dataset or MHCflurry EL, use another script. \n \
    This script can be used to explore data by giving allele name and peptide length. Please provide the -d directive if you want to create a csv. This script is used to generate the DB1. \n \
    The output csv file will have extended number of columns with the ID column (position 0) and the location of generated model in DB2 as the last column \n \
    The database can be generated using https://github.com/openvax/mhcflurry/tree/master/downloads-generation/data_curated curate.py script. \
    For a quick sight at the expected format: https://data.mendeley.com/datasets/zx3kjzc3yx/3 -> Data_S3.csv.\
    The output csv will have the same columns with the ID at the beggining and the path for the DB2 as the last one.\
    Csv file columns: allele, peptide, ba_value, equality/inequality, measurement_type, measurement_kind etc..\
    The measurement_type depends on the measurment_kind can be qualitative (elution, binding affinity) \
    or quantitative (binding_affinity).\
    ')
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

arg_parser.add_argument(
    "--peptide-length", "-P",
    help="Length of peptides.",
    default=0,
    type=int
)
arg_parser.add_argument(
    "--allele", "-A",
    default=[],
    help="Name of the allele(s) to filter. \n \
        More than one allele have to be separated by spaces: -A HLA-A HLA-B. \n \
        The allele name can be resolved up the the last position (HLA-A*02:01) or only for a specie (HLA). \n \
        If no allele is provided, every alleles will be returned.",
    nargs="+",
)
arg_parser.add_argument("--prefix", "-i",
    help="The prefix for the ID",
    choices=["EL", "BA"],
    required=True
)

a = arg_parser.parse_args();

# build the authorized alleles list based on arguments provided
rows = [];

input_csv_df = pd.read_csv(f"{a.source_csv}")
ids = list(range(len(input_csv_df)))
input_csv_df.insert(0, column="ID", value=[f"{a.prefix}_{id+1}" for id in ids]) # give an ID to each entry
# PANDORA generated models location (provided as an argument for the modeling, among peptide and MHC allele):
input_csv_df["db2_folder"] = [f"/projects/0/einf2380/data/pMHCI/models/{a.prefix}/{assign_outfolder(id+1)}" for id in ids]

# filter only discrete and quantitative measurements. This filter is applied for pilot study 
# as a pre-filder (before filtering alleles and peptide length):
input_csv_df = input_csv_df.query("measurement_inequality == '=' & measurement_type == 'quantitative' & \
    measurement_kind == 'affinity' & measurement_value >= 2")

# apply the allele and length of peptide filter:
output_csv_df = input_csv_df
if len(a.allele) > 0:
    allele_mask = [ # boolean mask to filter the df for target alleles 
        any(allele_filter in allele_id for allele_filter in a.allele) # return true if one of the a.allele string is in the allele_id string
        for allele_id in input_csv_df["allele"].tolist() # convert the allele of the df into a list
    ]
    output_csv_df = input_csv_df[allele_mask] # filter for the rows which have the specified alleles

if a.peptide_length > 0:
    # simple mask to filter rows which have the desired peptide length:
    peptide_mask = [len(peptide) == a.peptide_length for peptide in output_csv_df["peptide"].tolist()]
    output_csv_df = output_csv_df[peptide_mask]

#save the csv:
if a.output_csv:
    output_csv_df.to_csv(f"{a.output_csv}", index=False)
    print(f"file {a.output_csv} with {len(output_csv_df)} entries saved in data/external/processed/")
else:
    print(output_csv_df)
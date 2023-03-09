import argparse
import pandas as pd
from db1_to_db2_path import assign_outfolder

# list of arguments:
arg_parser = argparse.ArgumentParser(
    description='Step 2 of db1. Build a p:MHC Binding Affinity csv file. Optimized for MHCFlurry dataset. For NetMHCpan dataset or MHCflurry EL, use another script. \n \
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
    help="Name of the MHCflurry dataset csv file that has id's already added to it (with shell script 1_*.sh), usually found in: projectfolder/data/external/unprocessed.",
    default="../../data/external/unprocessed/BA_pMHCI_human_nonhuman.csv",
    #default="../../data/external/unprocessed/curated_training_data.no_additional_ms.csv"
)
arg_parser.add_argument(
    "--output-csv", "-d",
    help="Name of destination csv with filtered entries. If provided, will be saved. \
    Otherwise this script can be used to visualize the applied filters.",
    default = None
)
arg_parser.add_argument("--prefix", "-i",
    help="The prefix for the ID in the ouput path column of the csv. e.g. /projectfolder/3d_models/prefix",
    default="BA",
    required=True
)
arg_parser.add_argument(
    "--peptide-length", "-p",
    help="Length of peptides.",
    default=0,
    type=int
)
arg_parser.add_argument("--with-pseudosequence", "-w",
    help="""
    Appends a pseudosequence column. Useful for allele encoding for the MLP. Path to the 
    mhcflurry.allele_sequence.csv file should be provided as well.
    """,
    default=False,
    action="store_true"
)
arg_parser.add_argument("--pseudoseq-csv", "-W",
    help="""
    Path to the mhcflurry.allele_sequence.csv file.
    Default: /home/lepikhovd/3D-Vac/external/unprocessed/mhcflurry.allele_sequences.csv
    """,
    default="/home/lepikhovd/3D-Vac/external/unprocessed/mhcflurry.allele_sequences.csv"
)
arg_parser.add_argument(
    "--allele", "-a",
    default=[],
    help="Name of the allele(s) to filter. \n \
        More than one allele have to be separated by spaces: -A HLA-A HLA-B. \n \
        The allele name can be resolved up the the last position (HLA-A*02:01) or only for a specie (HLA). \n \
        If no allele is provided, every alleles will be returned.",
    nargs="+",
)
arg_parser.add_argument("--measurement-type", "-t",
    help='The type of measurement to filter from the input csv. Choices: "BA" or "MS"',
    choices=["MS", "BA"],
    required=True
)
arg_parser.add_argument("--include_inequality", "-e",
    help='flag to make sure measurements with inequalities (> or <) are also included',
    default=False,
    type=bool,
    action=argparse.BooleanOptionalAction
)
arg_parser.add_argument("--include_qualitative", "-q",
    help='flag to make sure qualitative measurements are also included',
    default=False,
    type=bool,
    action=argparse.BooleanOptionalAction
)

a = arg_parser.parse_args();

# PANDORA generated models location (provided as an argument for the modeling, among peptide and MHC allele):

input_csv_df = pd.read_csv(a.source_csv, header=0)
ids = input_csv_df['ID'].apply(lambda x: int(x.split('_')[1])).tolist()
input_csv_df["db2_folder"] = [f"/projects/0/einf2380/data/pMHCI/3d_models/{a.prefix}/{assign_outfolder(id+1)}" for id in ids]

# filter based on user input
query = []
if a.measurement_type == "BA":
    query.append("measurement_kind == 'affinity'")
else:
    query.append("measurement_kind == 'mass_spec'")

if not a.include_qualitative:
    query.append("measurement_type == 'quantitative'")
if not a.include_inequality:
    query.append("measurement_inequality == '='")

# construct the complete query
query_string = "&".join(query)

print(f"filtering the following: {query_string}")
# filter the dataframe
input_csv_df = input_csv_df.query(query_string)

#     # input_csv_df = input_csv_df.query("(measurement_inequality == '=' | (measurement_inequality == '>' & measurement_value > 499) | \
#     #  (measurement_inequality == '<' & measurement_value < 500)) & measurement_type == 'quantitative' & \
#     # measurement_kind == 'affinity' & measurement_value >= 2")

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

if a.with_pseudosequence:
    print("Adding pseudosequences column")
    pseudoseq_df = pd.read_csv(a.pseudoseq_csv)
    alleles = pseudoseq_df.allele.tolist()
    output_alleles =output_csv_df["allele"].tolist() 
    allele_pseudoseq_dict = dict(zip(pseudoseq_df.allele, pseudoseq_df.sequence))
    pseudosequences = []
    for allele in output_alleles:
        if allele in alleles:
            pseudosequence = allele_pseudoseq_dict[allele]
            pseudosequences.append(pseudosequence)
        else:
            pseudosequences.append("X"*37)
    output_csv_df["pseudosequence"] = pseudosequences

#save the csv:
if a.output_csv:
    output_csv_df.to_csv(f"{a.output_csv}", index=False)
    print(f"file {a.output_csv} with {len(output_csv_df)} entries saved in data/external/processed/")
else:
    print(output_csv_df)
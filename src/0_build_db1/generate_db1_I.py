import argparse
from write_to_model_csvs import assign_outfolder

# list of arguments:
arg_parser = argparse.ArgumentParser(
    description='Build a p:MHC Binding Affinity csv file for BA only with a precise value ("=" inequallity) from MHCFlurry dataset. For NetMHCpan dataset or MHCflurry EL, use another script. \n \
    This script can be used to explore data by giving allele name and peptide length. Please provide the -d directive if you want to create a csv. This script is used to generate the DB1. \n \
    The output csv file will have extended number of columns with the ID column (position 0) and the location of generated model in DB2 as the last column \n \
    The database can be generated using https://github.com/openvax/mhcflurry/tree/master/downloads-generation/data_curated curate.py script. \
    For a quick sight at the expected format: https://data.mendeley.com/datasets/zx3kjzc3yx/3 -> Data_S3.csv\
    ')
arg_parser.add_argument(
    "--source-csv", "-f",
    help="Name of the MHCflurry dataset csv file in data/external/unprocessed if different from the default.",
    default="UPDATE THIS AFTER PUTTING THE FILE ON SNELLIUS WITH SYMLINK!!!!!!"
)
arg_parser.add_argument(
    "--output-csv", "-d",
    help="Name of destination csv with filtered entries. Will be saved in data/external/processed",
    required=True
)

arg_parser.add_argument(
    "--peptide-length", "-P",
    help="Length of peptides.",
    default=0,
    type=int
)
arg_parser.add_argument(
    "--allele-column", "-a",
    default=1,
    help="Index of the column for alleles, if different from default (1)",
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
arg_parser.add_argument(
    "--peptide-column", "-p",
    default=2,
    help="Index of the column for peptides, if different from default (2)",
    type=int
)
arg_parser.add_argument("--prefix", "-p",
    help="The prefix for the ID",
    choices=["EL", "BA"],
    required=True
)

a = arg_parser.parse_args();

# build the authorized alleles list based on arguments provided
rows = [];

with open(f"../../data/external/unprocessed/{a.file_path}") as f:
    line_count = 0;
    for line in f:
        row = line.replace("\n","").split(",");
        if line_count == 0: # add the ID header and the outdir column
            row.insert(0, "ID")
            row.append("outdir")
            print(row)
        else:
            id = a.prefix +"_"+str(line_count);
            row.insert(0, id);
            out_folder = f"/projects/0/einf2380/data/pMHCI/models/{a.prefix}/{assign_outfolder(line_count)}"
            row.append(out_folder)
        line_count+=1
        rows.append(row)

for ln,row in enumerate(rows):
    if ln == 0: ln+=1;continue;
    allele_id = row[a.allele_column];
    peptide = row[a.peptide_column];
    value = int(float(row[3]))
    # toe = "affinity"
    # if value > 1 and toe == row[6]:
    if any(allele in allele_id for allele in a.allele) or a.allele == []:
        if len(peptide) == a.peptide_length:
            rows.append(row);
        elif a.peptide_length == 0:
            rows.append(row);
    ln+=1
## get the unique alleles of each human and all type of alleles:
alleles = set([row[1] for row in rows]);
print(f"unique alleles: {alleles}")
print(f"number of rows: {len(rows)}")

# create or update the csv file
if (a.destination_path):
    with open(a.destination_path, "w") as file:
        # ln = len([row for row in file])
        to_write = "\n".join([",".join(x) for x in rows])
        file.write(to_write)
    print(f"human csv file created, number of rows: {len(rows)}");
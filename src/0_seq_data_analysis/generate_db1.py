import sys 
import argparse

from sklearn.cluster import affinity_propagation
from write_to_model_csvs import assign_outfolder

arg_parser = argparse.ArgumentParser(description="Script used to rename the inital .csv file so the output can have an ID column \
    as well as a column for the output directory \n \
    id format: prefix + number (position in the csv file)");
arg_parser.add_argument("--file-path", "-f",
    help="path of the csv file",
    default= "/home/lepikhovd/3d-epipred/binding_data/quantitative_mhcI.csv",
    # required=True
)
arg_parser.add_argument("--prefix", "-p",
    help="The prefix for the ID",
    choices=["EL", "BA"],
    required=True
)
a = arg_parser.parse_args()

rows = [];

with open(a.file_path) as f:
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
    print("number of lines: ", line_count)
print("last row of the csv file:")
print(rows[-1])
to_write = "\n".join([",".join(row) for row in rows])
with open(a.file_path, "w") as file:
    file.write(to_write)
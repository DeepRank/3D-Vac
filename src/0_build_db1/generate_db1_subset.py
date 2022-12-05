import argparse
from statistics import mean

arg_parser = argparse.ArgumentParser(
    description='Extract subset from a given db1 csv file')
arg_parser.add_argument(
    "--input-db", "-i",
    help="Original database in csv format",
    required=True
)
arg_parser.add_argument(
    "--output-db", "-o",
    help="Output database",
    required=True
)
arg_parser.add_argument(
    "--peptide-length", "-l",
    help="Length of peptides.",
    #default=15,
    type=int
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
arg_parser.add_argument(
    "--duplicates", "-d",
    help="What to do with duplicated labels. Defaults to remove them",
    default = "remove",
    choices=['remove','mean','min','max', 'leave']
)

a = arg_parser.parse_args()

to_write = []
with open(a.input_db, 'r') as infile:
    for i, line in enumerate(infile):
        row = line.replace('\n','').split(',')
        if i == 0:
            header = row
        else:
            allele = row[header.index('allele')]
            pept = row[header.index('peptide')]
            if a.allele != [] and not any(allele == x for x in a.allele):
                continue
            if a.peptide_length and len(pept) != a.peptide_length:
                continue
            to_write.append(row)


# Identify duplicates
ids = []
occurencies = {(x[1], x[2]) : 0 for x in to_write}
for value in to_write:
    occurencies[(value[1], value[2])] += 1

doubles = [key for key in occurencies if occurencies[key] >1]
values = {double : [] for double in doubles}
for double in doubles:
    [values[double].append(float(x[4])) for x in to_write if (x[1], x[2]) == double]

if a.duplicates != 'leave':
    for i, value in enumerate(to_write):
        if a.duplicates == 'remove':
            if (value[1], value[2]) in doubles:
                to_write[i] = None
        elif a.duplicates == 'mean':
            to_write[i] = mean(values[(values[1], values[2])])
        elif a.duplicates == 'min':
            to_write[i] = min(values[(values[1], values[2])])
        elif a.duplicates == 'max':
            to_write[i] = max(values[(values[1], values[2])])
#print(values)
to_write = [x for x in to_write if x != None]

with open(a.output_db, 'w') as outfile:
    outfile.write((',').join(header) + '\n')
    for row in to_write:
        outfile.write((',').join(row) + '\n')


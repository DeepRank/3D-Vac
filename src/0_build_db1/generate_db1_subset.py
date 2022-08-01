import argparse

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

with open(a.output_db, 'w') as outfile:
    outfile.write((',').join(header) + '\n')
    for row in to_write:
        outfile.write((',').join(row) + '\n')


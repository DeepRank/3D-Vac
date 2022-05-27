import torch
import argparse

arg_parser = argparse.ArgumentParser(
    description="Fully connected layer to generate a model which predicts the binding affinity based on one-hot encoded \
    peptide sequence. Works only for a fixed length of 9 residues. Takes as input the csv file (header free) containing \
    the list of peptides, the column for the peptides and a threshold for binding affinity to define binders."
)

arg_parser.add_argument("--csv-file", "-f",
    help="Absolute path of the csv file",
    default="/home/lepikhovd/3d-epipred/binding_data/BA_pMHCI.csv"
)
arg_parser.add_argument("--peptide-column", "-p",
    type=int,
    help="Column index of peptide's sequence in the csv file",
    default=2
)
arg_parser.add_argument("--threshold", "-t",
    help="Binding affinity threshold to define binders, default 500",
    type=float,
    default=float(500)
)

a = arg_parser.parse_args()

# retrieve peptides from the csv file
with open(a.csv_file, "r") as csv_f:
    rows = [row.split(",") for row in csv_f]
    peptides = []
    labels = []
    for row in rows:
        peptides.append(row[2])
        labels.append((0,1)[float(row[3]) <= a.threshold])
positives = [label for label in labels if label == 1]
negatives = [label for label in labels if label == 0]
print(f"Number of binders: {len(positives)}")
print(f"Number of non binders: {len(negatives)}")
print(f"Binders and non binders combined: {len(negatives)+len(positives)}")


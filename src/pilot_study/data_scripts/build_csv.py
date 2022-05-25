from functools import total_ordering
import os
import pickle
from re import I
import sys
from matplotlib import pyplot as plt
from math import exp, log
import pandas as pd
import argparse

# list of arguments:
arg_parser = argparse.ArgumentParser(description='Build a p:MHC Binding Affinity csv file from MHCFlurry dataset. For NetMHCpan, use another script')
arg_parser.add_argument(
    "--species", "-s",
    choices=["human", "all"],
    help="filters for the allele selection, required", 
    required=True
)
arg_parser.add_argument(
    "--source-csv", "-f", 
    help="path to the file if different from the default", 
    default="/home/lepikhovd/binding_data/quantitative_mhcI.csv"
)
arg_parser.add_argument(
    "--destination-path", "-d", 
    help="path to the destination csv", 
    default="/home/lepikhovd/binding_data"
    # required=True
)

arg_parser.add_argument(
    "--peptide-length", "-P", 
    help="length of peptides to select",
    default=0,
    type=int
)
arg_parser.add_argument(
    "--allele-column", "-a", 
    default=1, 
    help="index of the column for alleles, if different from default (0)",
    type=int
)
arg_parser.add_argument(
    "--allele", "-A",
    default=None,
    help="name of the allele to filter"
)
arg_parser.add_argument(
    "--peptide-column", "-p", 
    default=2, 
    help="index of the column for peptides, if different from default (1)",
    type=int
)
arg_parser.add_argument(
    "--toe", 
    choices=["EL", "BA"], 
    help="type of experiment, either BA or EL (only for id, not used as filter)", 
    required=True)

a = arg_parser.parse_args();

# build the authorized alleles list based on arguments provided
authorized_alleles = ["HLA-A"]
rows = []
ln = 0
with open(a.source_csv, 'r') as infile:
    for line in infile:
        if ln == 0: ln+=1;continue;
        row = line.replace("\n","").split(',')
        allele_id = row[a.allele_column];
        peptide = row[a.peptide_column];
        value = int(float(row[3]))
        toe = ("affinity", "mass_spec")[a.toe == "EL"]
        if value > 1 and toe == row[6] and row[4] == "=":
            if a.species == "human" and allele_id == "HLA-A*02:01":
                if len(peptide) == a.peptide_length:
                    rows.append(row);
                elif a.peptide_length == 0:
                    rows.append(row);
            elif a.species == "all":
                if len(peptide) == a.peptide_length:
                    rows.append(row);
                elif a.peptide_length == 0:
                    rows.append(row);
        ln+=1
## get the unique alleles of each human and all type of alleles:
alleles = set([row[1] for row in rows]);
print(f"number of unique alleles in {a.species} species: {len(alleles)}")

#create or update the csv file
with open(f"{a.destination_path}/{a.toe}_pMHCI.csv", "w") as file:
    # ln = len([row for row in file])
    to_write = "\n".join([",".join(x) for x in rows])
    file.write(to_write)
print(f"human csv file created, number of rows: {len(rows)}");

# with open("/home/daniill/netmhc_training_dataset/mhc2_ba_el.csv", "w") as file:
#     file.write("\n".join([",".join(x) for x in all_rows]))
# print(f"all species mhc2 csv file created, number of rows: {len(all_rows)}");

# with open("/home/daniill/netmhc_training_dataset/alleles.pkl", "rb") as f:
#     data = pickle.load(f)
#     human_alleles = data[0];
#     all_alleles = data[1];
#     human_rows = data[2];
#     all_rows = data[3];
# print(f"human alleles: {len(human_alleles)}")
# print(f"all alleles: {len(all_alleles)}")
# print(f"human rows: {len(human_rows)}")
# print(f"all rows: {len(all_rows)}")

# ###Preapare data for pie chart
# ba_neg_data = [row for row in all_rows if row[-1] == "BA" and exp((float(row[1])) * log(50000)) > 500];
# ba_pos_data = [row for row in all_rows if row[-1] == "BA" and exp((float(row[1])) * log(50000)) < 500];
# el_pos_data = [row for row in all_rows if row[-1] == "EL" and int(row[1]) == 1];
# el_neg_data = [row for row in all_rows if row[-1] == "EL" and int(row[1]) == 0];

# h_ba_neg_data = [row for row in human_rows if row[-1] == "BA" and exp((float(row[1])) * log(50000)) > 500];
# h_ba_pos_data = [row for row in human_rows if row[-1] == "BA" and exp((float(row[1])) * log(50000)) < 500];
# h_el_pos_data = [row for row in human_rows if row[-1] == "EL" and int(row[1]) == 1];
# h_el_neg_data = [row for row in human_rows if row[-1] == "EL" and int(row[1]) == 0];

# ba_neg = len(ba_neg_data)
# ba_pos = len(ba_pos_data)
# el_pos = len(el_pos_data)
# el_neg = len(el_neg_data)

# h_ba_neg = len(h_ba_neg_data)
# h_ba_pos = len(h_ba_pos_data)
# h_el_pos = len(h_el_pos_data)
# h_el_neg = len(h_el_neg_data)


# ## make the human pie chart
# tot_len = h_ba_neg + h_ba_pos + h_el_neg + h_el_pos

# plt.pie([(h_el_pos + h_el_neg)/tot_len*100,(h_ba_neg+ h_ba_pos)/tot_len*100], 
#     labels=['EL: %i '%(h_el_neg + h_el_pos), 'BA %i'%(h_ba_neg + h_ba_pos)], autopct='%1.1f%%')
# plt.title('Total: ' + str(tot_len) + ' experimental values for both BA and EL \n data for mono-allelic experiments in human') 
# plt.savefig("/home/daniill/human_MHC2_plot.png")
# plt.clf()
# print("first plot (human) ploted")

# plt.pie([h_el_pos, h_el_neg, h_ba_pos, h_ba_neg], 
#        labels = ['EL pos: %i' %h_el_pos, 'EL neg: %i' %h_el_neg,
#         'BA pos: %i' %h_ba_pos, 'BA neg: %i' %h_ba_neg], autopct='%1.1f%%')
# plt.title('Proportion of negative and positive entries for both \n BA and EL data for mono-allelic experiments in human') 
# plt.savefig("/home/daniill/human_pos_neg_MHC2_plot.png")
# plt.clf()
# print("second plot (human) plotted")

# ## make multi-species pie chart

# tot_len = ba_neg + ba_pos + el_pos + el_neg

# plt.pie([(el_pos + el_neg)/tot_len*100,(ba_pos + ba_neg)/tot_len*100], 
#     labels=['EL: %i '%(el_neg + el_pos), 'BA %i'%(ba_pos + ba_neg)], autopct='%1.1f%%')
# plt.title('Total: ' + str(tot_len) + ' experimental values for both BA and EL \n data for mono-allelic experiments in different species') 
# plt.savefig("/home/daniill/all_species_MHC2_plot.png")
# plt.clf()
# print("first (all species) plot ploted")

# plt.pie([el_pos, el_neg, ba_pos, ba_neg], 
#        labels = ['EL pos: %i' %el_pos, 'EL neg: %i' %el_neg,
#         'BA pos: %i' %ba_pos, 'BA neg: %i' %ba_neg], autopct='%1.1f%%')
# plt.title('Positive and negative mono-allelic entries from the BA and EL data for several species') 
# plt.savefig("/home/daniill/all_species_pos_neg_MHC2_plot.png")
# plt.clf()
# print("second (all species) plot plotted")

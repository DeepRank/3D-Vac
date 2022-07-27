import argparse
import pandas as pd
from db1_to_db2_path import assign_outfolder
import os
from math import exp, log

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
    "--source-folder", "-f",
    help="path to the NetMHCIIPan4.0 dataset in data/external/unprocessed if different from the default.",
    default="../../data/external/unprocessed/pMHCII/"
)
arg_parser.add_argument(
    "--output-csv", "-d",
    help="Name of destination csv with filtered entries. \
    Otherwise this script can be used to visualize the applied filters.",
    default = None
)

# arg_parser.add_argument(
#     "--peptide-length", "-P",
#     help="Length of peptides.",
#     #default=15,
#     type=int
# )
# arg_parser.add_argument(
#     "--allele", "-A",
#     default=[],
#     help="Name of the allele(s) to filter. \n \
#         More than one allele have to be separated by spaces: -A HLA-A HLA-B. \n \
#         The allele name can be resolved up the the last position (HLA-A*02:01) or only for a specie (HLA). \n \
#         If no allele is provided, every alleles will be returned.",
#     nargs="+",
# )
arg_parser.add_argument("--prefix", "-i",
    help="The prefix for the ID",
    choices=["EL", "BA"],
    default='BA'
)


def assign_outfolder(index):
    '''Assigns outfolder name depending on case ID
    
    Args:
        index (int): case ID
        
    Returns:
        interval (str): folder name definying the interval of cases 
                        the folder will contain
    '''
    
    if index%1000 != 0:
        interval = '%i_%i' %((int(index/1000)*1000)+1, (int(index/1000)*1000)+1000)
    else:
        interval = '%i_%i' %(index-999, index)
    return interval
        
def adapt_allele_name(allele):
    """ Adapts the allele name from the syntax in file to the standard syntax

    Args:
        allele (str): Allele name in original dataset file

    Returns:
        allele (str): Standard syntax allele name
    """    
    #TODO: to extend for other alleles
    if allele.startswith('DRB'):
        allele = allele.replace('_', '*')
        allele = allele[:-2] + ':' + allele[-2:]
        allele = 'HLA-'+ allele
        
    return allele

if __name__ == "__main__":
    a = arg_parser.parse_args()
    # build the authorized alleles list based on arguments provided
    # rows = []

    # input_csv_df = pd.read_csv(f"../../data/external/unprocessed/{a.source_csv}")
    # ids = list(range(len(input_csv_df)))
    # input_csv_df["ID"] = [f"{a.prefix}_{id+1}" for id in ids] # give an ID to each entry
    # # PANDORA generated models location (provided as an argument for the modeling, among peptide and MHC allele):
    # input_csv_df["db2_folder"] = [f"/projects/0/einf2380/data/pMHCI/models/{a.prefix}/{assign_outfolder(id+1)}" for id in ids]

    # filter only discrete and quantitative measurements. This filter is applied for pilot study 
    # as a pre-filder (before filtering alleles and peptide length):
    # input_csv_df = input_csv_df.query("measurement_inequality == '=' & measurement_type == 'quantitative' & \
    #     measurement_kind == 'affinity' & measurement_value >= 2")

    # apply the allele and length of peptide filter:
    # output_csv_df = input_csv_df
    # if len(a.allele) > 0:
    #     allele_mask = [ # boolean mask to filter the df for target alleles 
    #         any(allele_filter in allele_id for allele_filter in a.allele) # return true if one of the a.allele string is in the allele_id string
    #         for allele_id in input_csv_df["allele"].tolist() # convert the allele of the df into a list
    #     ]
    #     output_csv_df = input_csv_df[allele_mask] # filter for the rows which have the specified alleles

    # if a.peptide_length > 0:
    #     # simple mask to filter rows which have the desired peptide length:
    #     peptide_mask = [len(peptide) == a.peptide_length for peptide in output_csv_df["peptide"].tolist()]
    #     output_csv_df = output_csv_df[peptide_mask]

    # #save the csv:
    # if a.output_csv:
    #     output_csv_df.to_csv(f"../../data/external/processed/{a.output_csv}", index=False)
    #     print(f"file {a.output_csv} with {len(output_csv_df)} entries saved in data/external/processed/")
    # else:
    #     print(output_csv_df)

######################
    #%% 
    #Get a list of acceptable allele names (to differentate them from cell lines)
    cell_to_alleles = {}
    with open(a.source_folder + 'allelelist.txt') as infile:
        for line in infile:
            cell_to_alleles[line.split(' ')[0]] = line.split(' ')[1].replace('\n', '').split(',')
    acceptable_ids = {x:cell_to_alleles[x] for x in cell_to_alleles if len(cell_to_alleles[x]) == 1}
    
    ##Get NetMHCIIPan4.1 (NMP2) training data
    NMP2_data_folder = a.source_folder
    ba_values = []
    for f in os.listdir(NMP2_data_folder):
        if not f.startswith('allelelist') and a.prefix + '1' in f and f.endswith('.txt'):
            print(f)
            with open(NMP2_data_folder + f, 'r') as infile:
                for line in infile:
                    row = line.split('\t')
                    affinity = exp((float(row[1])) * log(50000))
                    if row[2] in list(acceptable_ids.keys()):
                        ba_values.append([row[0], row[2].replace('\n', ''), float(row[1]), affinity]) 

    #%% Remove ba redundancies
    ba_values = set(tuple(x) for x in ba_values)
    ba_values = sorted(list(ba_values), key=lambda x:x[0])

    #%% Make Binding Affinity dictionary
    ba_dict = {}
    for x in ba_values:
        #if 'HLA-' in x[1] and '/' not in x[1]:67
        try:
            ba_dict[x[0]][x[1]].append(x[2])
        except KeyError:
            ba_dict[x[0]] = {x[1]:[x[2]]}

    # Pickle the data
    # with open('../binding_data/pMHCII/ba_values.pkl', 'rb') as inpkl:
    #     ba_values = pickle.load(inpkl)
    #     ba_dict = pickle.load(inpkl)

    #Define output folder
    MHCII_outfolder = '/projects/0/einf2380/data/pMHCII/models/'
    if a.prefix == 'BA':
        outfolder = MHCII_outfolder + 'BA/'
    elif a.prefix == 'EL':
        outfolder = MHCII_outfolder + 'EL/'

    #%% Write output csv
    #with open('../binding_data/pMHCII/IDs_BA_MHCII.csv', 'w') as outcsv:
    with open(a.output_csv, 'w') as outcsv:
        header = ['ID', 'peptide', 'allele', 
                'score', 'measurment', 'output_folder']
        outcsv.write((',').join(header) + '\n')
        for i, case in enumerate(ba_values):
            index = i + 1
            ID = '%s_%i' %(a.prefix, index)
            pept = case[0]
            allele = adapt_allele_name(case[1])
            score = str(case[2])
            ba = str(case[3])
            out_interval = assign_outfolder(index)
            case_outfolder = outfolder + out_interval
            row = [ID, pept, allele, score, ba, case_outfolder]
            #outcsv.write((',').join(row))
            #if allele == 'HLA-DRB1*01:01':
            outcsv.write((',').join(row) + '\n')
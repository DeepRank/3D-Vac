import argparse
import pandas as pd
from db1_to_db2_path import assign_outfolder
import os
from math import exp, log

# list of arguments:
arg_parser = argparse.ArgumentParser(
    description='Build a p:MHC Binding Affinity csv file for BA from NetMHCIIpan4.1 database. \n \
    The output csv file will have extended number of columns with the ID column (position 0) and the location of generated model in DB2 as the last column \n \
    The output csv will have the same columns with the ID at the beggining and the path for the DB2 as the last one.\
    Csv file columns: allele, peptide, score, measurement,etc..\
    ')
arg_parser.add_argument(
    "--source-folder", "-f",
    help="path to the NetMHCIIPan4.0 dataset in data/external/unprocessed if different from the default.",
    default="../../data/external/unprocessed/pMHCII/"
)
arg_parser.add_argument(
    "--output-csv", "-o",
    help="Name of destination csv with filtered entries. \
    Otherwise this script can be used to visualize the applied filters.",
    default = None
)
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
    elif allele.startswith('HLA') and len(allele.split('-')) == 3:
        allele = allele.split('-')[1:]
        allele = [f'HLA-{x[:4]}*{x[4:6]}:{x[6:]}' for x in allele]
        allele = (';').join(allele)
        #DQA10401
    elif allele.startswith('H-'):
        pass
    else:
        print(f'Unrecognized allele syntax: {allele}')

    return allele

if __name__ == "__main__":
    a = arg_parser.parse_args()
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

    #Define output folder
    MHCII_outfolder = '/projects/0/einf2380/data/pMHCII/3D_models/'
    if a.prefix == 'BA':
        outfolder = MHCII_outfolder + 'BA/'
    elif a.prefix == 'EL':
        outfolder = MHCII_outfolder + 'EL/'

    #%% Write output csv
    #with open('../binding_data/pMHCII/IDs_BA_MHCII.csv', 'w') as outcsv:
    with open(a.output_csv, 'w') as outcsv:
        header = ['ID', 'allele', 'peptide', 
                'score', 'measurment', 'db2_folder']
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
            row = [ID, allele, pept, score, ba, case_outfolder]
            #outcsv.write((',').join(row))
            #if allele == 'HLA-DRB1*01:01':
            outcsv.write((',').join(row) + '\n')
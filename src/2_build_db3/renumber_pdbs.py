import random
import glob
import os
import math
import traceback
import argparse
import subprocess
import numpy as np
import pandas as pd
from joblib import Parallel, delayed 

parser = argparse.ArgumentParser(
    "Renumber pdbs so that amino acid group numbering is consistent across alleles, this is useful for the \
    alignment of the pdbs because this process relies on these specific indices. A binary of the MUSCLE software (https://www.drive5.com/muscle/) \
    is used to perform a more reliable alignment of the amino acid sequences per allele, the corresponding alignment is used to determine which \
    positions are indels. With this information the amino acid groups are consistently renumbered")

parser.add_argument('-f',
                    '--folder',
                    type=str,
                    help='The directory containing the pdbs')
parser.add_argument('-c',
                    '--csv',
                    type=str,
                    required= False,
                    help='A csv to filter the pdbs that need to be renumbered instead of renumbering everything from db3')
parser.add_argument('-n',
                    '--n-cores',
                    type=int,
                    help='Number of cores available (specify the same number in sbatch script if applicable')


aaDict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q',
          'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
          'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

# Align sequences
def muscle(fasta_in, fasta_out):
    try:
        output = subprocess.run(f"muscle -align {fasta_in} -output {fasta_out}", shell=True, 
                                check=True, capture_output=True)
        print(f"Muscle output: {output.stdout.decode('utf-8')}")
    except subprocess.CalledProcessError as cpe:
        print(cpe)

def renumber(pdb_fasta_list:list, chainId='M'):
    """renumber the pdbs with consistent numbering of the amino acid groups
       per allele
    Args:
        pdb_fasta_list (list): an array with corresponding pairs: (pdb_path, fasta_string)
        chainId (str, optional): The chain of the pdb to renumber. Defaults to 'M'.
    """
     
    for item in pdb_fasta_list:
        try:   
            pdbFile, fastaline = item
            pdb = open(pdbFile).read().split('\n')
            with open(pdbFile, 'w') as pdb2:
                nmb2 = []
                for i in range(len(fastaline)):
                    if fastaline[i] != '-':
                        nmb2 += [i+1]
                tel = -1
                his = '-1'
                for line in pdb:
                    if line.startswith('ATOM ') and line[21] == chainId:
                        if line[22:27] != his:
                            tel += 1
                            his = line[22:27]
                        pdb2.write(line[:22] + (' '*(4-len(str(nmb2[tel])))
                                                ) + str(nmb2[tel]) + " "+line[27:]+'\n')
                    elif line.startswith('TER '):
                        pdb2.write(line[:22] + (' '*(4-len(str(nmb2[tel])))
                                                ) + str(nmb2[tel])+'\n')
                    else:
                        pdb2.write(line+'\n')
        except Exception as e:
            print(f'Failed to process pdb {pdbFile} with fasta sequence {fastaline}\nline: {line}\nnmb2: {nmb2}\n tel: {tel}')
            print(f'{e}\n{traceback.format_exc()}')


def getPdbs(folder):
    pdbs_list = glob.glob(os.path.join(folder, '*'))
    # Sort the files alphabetically
    pdbs_list.sort(key=str.lower)

    return pdbs_list


def getSequence(pdb:str, chain="M"):
    """get the amino acid sequence from the input pdb

    Args:
        pdb (str): path of pdb file
        chain (str, optional): chain descriptor in the pdb, only get the atoms 
                                   linked to that chain Defaults to "M" (=MHC).

    Returns:
        str: sequence of protein chain
    """    
    pdb = open(pdb).read().split('\n')
    pdb = [line[17:27]
           for line in pdb if line.startswith('ATOM ') and line[21] == chain]
    pdb2 = []
    his = '-1'
    for i in pdb:
        if i != his:
            pdb2 += [aaDict[i[:3]]]
            his = i
    return ''.join(pdb2)


def readFasta(fname):
    return [[i.split('\n')[0], ''.join(i.split('\n')[1:])] for i in open(fname).read().split('>')[1:]]

def get_unique_sequences(pdbs:list, pdbs_complete: list):
    """pair the unique protein sequences (found after parsing all the models) with a list of indices 
    corresponding to the posititon of the pdb paths (that match with that sequence) in a fixed list of paths (pdbs_complete)
    example: {'MCGVALG..':[2,23,189], 'MIKCAP..': [67, 290, 3489], ..}

    Args:
        pdbs (list): a sublist of pdbs_complete (list was chunked in order to parallelize this process)
        pdbs_complete (list): complete list of paths of all the pdbs

    Returns:
        dic (dict): dictionary containing a mapping of the unique sequences with all the indices referring to pdb paths
    """    
    sequences = [getSequence(pdb) for pdb in pdbs]
    dic = {}
    for i, (seq, pdbfile) in enumerate(zip(sequences, pdbs)):
        try:
            dic[seq] += [pdbs_complete.index(pdbfile)]
        except:
            dic[seq] = [pdbs_complete.index(pdbfile)]
    return dic

def combine_dicts(list_dicts: list):
    """combine dictionaries containing indices of pdbs files as items and unique sequences as keys
    Args:
        list_dicts (list): the dictionaries produced in parallel in need of combining

    Returns:
        dict: the combined dictionary
    """    
    combined_dict = {}
    for pdb_dict in list_dicts:
        for seq, index_pdb in pdb_dict.items():
            try:
                combined_dict[seq] = list(set(combined_dict[seq] + index_pdb))
            except:
                combined_dict[seq] = index_pdb
    return combined_dict

def fast_load_dirs(sub_folder):
    pdb_files = glob.glob(os.path.join(sub_folder, '*/pdb/*.pdb'))    
    return pdb_files

def main(folder, csv, n_cores):
    # if csv is provided
    if csv:
        df = pd.read_csv(csv, header=0)
        filter_ids = df['ID'].tolist()
    else:
        filter_ids = False
    # this yields a list of all the sub folders
    pdb_subs = glob.glob(os.path.join(folder, '*'))
    
    # load all the pdbs in parallel manner
    pdbs_list = Parallel(n_jobs=n_cores, verbose=1)(delayed(fast_load_dirs)(pdb_sub) for pdb_sub in pdb_subs)
    pdbs_complete = [x for sublist in pdbs_list for x in sublist]
    print(f'number of pdbs: {len(pdbs_complete)}')
    
    # get unique amino acid sequence from pdb, let each inner list be handled by exactly one thread
    lists_pdb_dicts = Parallel(n_jobs=n_cores, verbose=1)(delayed(get_unique_sequences)(path_list, pdbs_complete) for path_list in np.array_split(pdbs_complete, n_cores))

    # get the combined dictionary of all the parallel processes
    combined_pdb_dict = combine_dicts(lists_pdb_dicts)
    
    fasta_in = os.path.join('/projects/0/einf2380/data/temp/renumber_pdbs', 'tmp_fasta.fa')
    fasta_out = os.path.join('/projects/0/einf2380/data/temp/renumber_pdbs', 'tmp_fasta_out.fa')
    
    with open(fasta_in, 'w') as fastafile:
        [fastafile.write('>%s\n%s\n' % (seq, seq)) for seq in combined_pdb_dict.keys()]
    
    # perform the alignment
    muscle(fasta_in, fasta_out)

    aligned = readFasta(fasta_out)
    pdb_new_alignment_pair = []
    
    # combine the dictionaries and put the corresponding (pdb, sequence) pairs in a list
    for unique_seq, alignment in aligned:
        corresponding = combined_pdb_dict[unique_seq]
        for pdb_ind in corresponding:
            if filter_ids and os.path.basename(pdbs_complete[pdb_ind]).split('.')[0] in filter_ids:
                pdb_new_alignment_pair.append([pdbs_complete[pdb_ind], alignment])
            elif not filter_ids:
                pdb_new_alignment_pair.append([pdbs_complete[pdb_ind], alignment])
    print(f'Number of pdbs to be renamed {len(pdb_new_alignment_pair)}')
    
    # perform filtering if needed 
    if filter_ids:
        pdb_new_alignment_pair = [[pdb,seq] for pdb, seq in zip(pdb_new_alignment_pair) if os.path.basename(pdbs_complete[pdb_ind]).split('.')[0] in filter_ids]
    #perform the renumbering in parallel
    Parallel(n_jobs = n_cores, verbose=1)(delayed(renumber)(path_list) for path_list in np.array_split(pdb_new_alignment_pair, n_cores))
    print('done!')


if __name__ == '__main__':
    a = parser.parse_args()
    main(a.folder, a.csv, a.n_cores)

import random
import glob
import os
import math
import pickle
import traceback
import argparse
import subprocess
import numpy as np
from joblib import Parallel, delayed 

parser = argparse.ArgumentParser(
    "renumber pdbs so that positions are consistent across all cases")

parser.add_argument('-f',
                    '--folder',
                    type=str,
                    help='The directory containing the pdbs')
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

# Renumber pdb-file
def renumber(pdb_fasta_list:list, chainId='M'):
    """renumber 

    Args:
        pdb_fasta_list (list): _description_
        chainId (str, optional): _description_. Defaults to 'M'.
    """
     
    for item in pdb_fasta_list:
        try:   
            pdbFile, fastaline = item
            pdb = open(pdbFile).read().split('\n')
            with open(pdb, 'w') as pdb2:
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
                print(f"processed pdb successfully {pdbFile}")
        except Exception as e:
            print(f'Failed to process pdb {pdbFile} with fasta sequence {fastaline}\nline: {line}\nnmb2: {nmb2}\n tel: {tel}')
            print(f'{e}\n{traceback.format_exc()}')


def getPdbs(folder):
    #  pdbs_list = []
    # if 1:
    #     for root, _, files in os.walk(folder):
    #         for file in [file for file in files if file.endswith('.pdb')]:
    #             pdbs_list.append(str(os.path.join(root, file)))

    pdbs_list = glob.glob(os.path.join(folder, '*'))
    # Sort the files alphabetically
    pdbs_list.sort(key=str.lower)

    return pdbs_list


def getSequence(pdb, chain="M"):
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


# def create_output_folder(pdbspath):
#     basedir = os.path.split(os.path.dirname(pdbspath))[0]
#     newpath = os.path.join(basedir, 'pdb_renumbered2')
#     if not os.path.exists(newpath):
#         os.mkdir(newpath)
#     else:
#         pass
#         # print(f'removing files from directory {newpath} if they exist')
#         # try:
#         #     subprocess.run(f'rm -r {newpath}/*', shell=True, check=True)
#         # except subprocess.CalledProcessError as cpe:
#         #     print(cpe)
#         # print(f'done removing files')
#     return newpath

def fast_load_dirs(sub_folder):
    pdb_files = glob.glob(os.path.join(sub_folder, '*/pdb/*.pdb'))    
    return pdb_files

def main(folder, n_cores):
    # get all the pdb paths
    # pdbs_complete = getPdbs(folder)
    # this yields a list of all the sub folders
    pdb_subs = glob.glob(os.path.join(folder, '*'))
    
    pdbs_list = Parallel(n_jobs=n_cores, verbose=1)(delayed(fast_load_dirs)(pdb_sub) for pdb_sub in pdb_subs)
    pdbs_complete = [x for sublist in pdbs_list for x in sublist]
    # let each inner list be handled by exactly one thread
    lists_pdb_dicts = Parallel(n_jobs=n_cores, verbose=1)(delayed(get_unique_sequences)(path_list, pdbs_complete) for path_list in np.array_split(pdbs_complete, n_cores))
    
    with open('list_pdb_dicts.pkl', 'wb') as writefile:
        pickle.dump(lists_pdb_dicts, writefile)
    # reduce list of lists to a (1d) list
    # list_pdbs_dicts = list(np.array(lists_pdb_dicts).flat)
    # get the combined dictionary of all the parallel processes
    combined_pdb_dict = combine_dicts(lists_pdb_dicts)
    
    fasta_in = os.path.join('/projects/0/einf2380/data/temp/renumber_pdbs', 'tmp_fasta.fa')
    fasta_out = os.path.join('/projects/0/einf2380/data/temp/renumber_pdbs', 'tmp_fasta_out.fa')
    # fasta_in = os.path.join('/home/severin/teststuff/test_renaming', 'tmp_fasta.fa')
    # fasta_out = os.path.join('/home/severin/teststuff/test_renaming', 'tmp_fasta_out.fa')
    
    with open(fasta_in, 'w') as fastafile:
        [fastafile.write('>%s\n%s\n' % (seq, seq)) for seq in combined_pdb_dict.keys()]
    
    muscle(fasta_in, fasta_out)

    aligned = readFasta(fasta_out)
    pdb_new_alignment_pair = []
    for unique_seq, alignment in aligned:
        corresponding = combined_pdb_dict[unique_seq]
        for pdb_ind in corresponding:
            pdb_new_alignment_pair.append([pdbs_complete[pdb_ind], alignment])
    print(f"len pdb_new_alignment_pair: {pdb_new_alignment_pair}")
    
    with open('pdb_new_alignment_pair.pkl', 'wb') as outfile:
        pickle.dump(pdb_new_alignment_pair, outfile)
    
    with open('combined_pdb_dict.pkl', 'wb') as outfile:
        pickle.dump(combined_pdb_dict, outfile)
            # renumber(pdbs[pdb_ind], alignment, 'M')
    Parallel(n_jobs = n_cores, verbose=1)(delayed(renumber)(path_list) for path_list in np.array_split(pdb_new_alignment_pair, n_cores))
    print('done!')


if __name__ == '__main__':
    a = parser.parse_args()
    # renumbered_path = create_output_folder(a.folder)
    main(a.folder, a.n_cores)

import glob
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import os
import pdb2sql
import argparse

"""
Usage: srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --time=00:30:00 python check_step2_completion.py
"""
arg_parser = argparse.ArgumentParser(
    description="""Checks if step2 has been propely performed.
    """
)
arg_parser.add_argument("--db2-path", "-d",
    help="Path to db2",
    type=str,
    required=True
)
arg_parser.add_argument("--ids-csv", "-i",
    help="Path to csv with IDs",
    type=str,
    required=True
)
arg_parser.add_argument("--output-csv", "-o",
    help="Path to output file",
    type=str,
    required=True
)
arg_parser.add_argument("--mhc-class", "-m",
    help="MHC class",
    type=str,
    choices=['I','II']
)
arg_parser.add_argument("--n-jobs", "-n",
    help="number of jobs",
    type=int,
    default=128
)


aminoacids = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def check_case(paths, IDs_csv, mhc):
    checks = {x: None for x in ['pdb','pdb_M','M_len',
                'pdb_P','P_len', 'M_pssm', 'M_pssm_as_pdb',
                'P_pssm', 'P_pssm_as_pdb', 'P_chain_all_same']}
    #CHECKS:
    IDs_list = []
    checks_list = []
    for path in paths:
        ID = ('-').join(path.split('/')[-1].split('-')[:2])
        #pdb is in there
        pdb_file = f'{path}/pdb/{ID}.pdb'
        if os.path.isfile(pdb_file):
            checks['pdb'] = True
        else:
            checks['pdb'] = False
            checks_list.append(checks)
            IDs_list.append(ID)
            continue

        #load data
        sql = pdb2sql.pdb2sql(pdb_file)

        #there is chain M
        if 'M' in sql.get_chains():
            checks['pdb_M'] = True
        else:
            checks['pdb_M'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        #chain M has 170<x<184 res (store residues)
        M_length = len(set(sql.get('resSeq', chainID='M')))
        if mhc == 'II' and 170<=M_length<=184:
            checks['M_len'] = M_length
        elif mhc == 'I' and 170<=M_length<=188:
            
            checks['M_len'] = M_length
        else:
            checks['M_len'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        #there is chain P
        if 'P' in sql.get_chains():
            checks['pdb_P'] = True
        else:
            checks['pdb_P'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        #chain P has 7<x<25 res (store residues)
        P_length = len(set(sql.get('resSeq', chainID='P')))
        if mhc == 'II' and 7<=P_length<=25:
            checks['P_len'] = P_length
        elif mhc == 'I' and 7<=P_length<=15:
            checks['P_len'] = P_length
        else:
            checks['P_len'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        #there is NO chain N
        if mhc == 'I':
            checks['no_N'] = False
        elif mhc == 'II' and 'N' not in sql.get_chains():
            checks['no_N'] = True
        else:
            checks['no_N'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        # M.pssm is in there
        M_pssm_file = f'{path}/pssm/{ID}.M.pdb.pssm'
        if os.path.isfile(M_pssm_file):
            checks['M_pssm']=True
        else:
            checks['M_pssm'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        #has same n residues and numbering as pdb
        with open(M_pssm_file) as Mpssm:
            next(Mpssm)
            resIDs = [int(line.split()[0]) for line in Mpssm]
        if sorted(resIDs) == sorted(list(set(sql.get('resSeq', chainID='M')))):
            checks['M_pssm_as_pdb'] = True
        else:
            checks['M_pssm_as_pdb'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        # P.pssm is in there
        P_pssm_file = f'{path}/pssm/{ID}.P.pdb.pssm'
        if os.path.isfile(P_pssm_file):
            checks['P_pssm']=True
        else:
            checks['P_pssm'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        #has same n residues and numbering as pdb
        with open(P_pssm_file) as Ppssm:
            next(Ppssm)
            resIDs = []
            pssm_P_seq = ''
            for line in Ppssm:
                row = line.split()
                resIDs.append(int(row[0]))
                pssm_P_seq += row[1]
        if sorted(resIDs) == sorted(list(set(sql.get('resSeq', chainID='P')))):
            checks['P_pssm_as_pdb'] = True
        else:
            checks['P_pssm_as_pdb'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue
        #ID, P chain lenght an d seq correspond to csv data
        pdb_P_seq = sorted(list(set([tuple(x) for x in sql.get('resSeq,resName', chainID='P')])))
        pdb_P_seq = ('').join([aminoacids[aa[1]] for aa in pdb_P_seq])
        if pdb_P_seq == pssm_P_seq == df[df['ID']==ID]['peptide'].item():
            checks['P_chain_all_same'] = True
        else:
            checks['P_chain_all_same'] = False
            IDs_list.append(ID)
            checks_list.append(checks)
            continue

        IDs_list.append(ID)
        checks_list.append(checks)

    return IDs_list, checks_list



if __name__=='__main__':
    n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))
    args = arg_parser.parse_args()
    
    # mhc class
    mhc = args.mhc_class 

    #db2_path = '/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/*/*'
    db2_path = args.db2_path.replace('\\', '')
    all_paths = glob.glob(db2_path)

    print(f'Total number of pdb paths {len(all_paths)}')

    #IDs_csv = '/projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv'
    df = pd.read_csv(args.ids_csv)

    ids, checks = zip(*Parallel(n_jobs = n_cores, verbose = 1)(delayed(check_case)(paths, df, mhc) for paths in np.array_split(all_paths, n_cores)))
    # print(checks)
    # checks = [{ID: check} for sublist in checks for (ID, check) in sublist]
    checks = {id: check for chunk_id, chunk_check in zip(ids, checks) for id, check in zip(chunk_id, chunk_check)}
    #checks = {x[0] : x[1] for x in checks}

    with open(args.output_csv, 'w') as outfile:
        header = ['ID'] + list(list(checks.values())[0].keys())
        outfile.write(('\t').join(header) + '\n')
        for key in checks:
            row = [key] + [str(x) for x in list(checks[key].values())]
            outfile.write(('\t').join(row) + '\n')
import glob
from joblib import Parallel, delayed
import pandas as pd
import os
import pdb2sql

"""
Usage: srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --time=00:30:00 python check_step2_completion.py
"""

aminoacids = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def check_case(path, IDs_csv):
    checks = {x: None for x in ['pdb','pdb_M','M_len',
                'pdb_P','P_len', 'M_pssm', 'M_pssm_as_pdb',
                'P_pssm', 'P_pssm_as_pdb', 'P_chain_all_same']}
    #CHECKS:
    ID = ('_').join(path.split('/')[-1].split('_')[:2])
    #pdb is in there
    pdb_file = f'{path}/pdb/{ID}.pdb'
    if os.path.isfile(pdb_file):
        checks['pdb'] = True
    else:
        checks['pdb'] = False
        return (ID, checks)

    #load data
    sql = pdb2sql.pdb2sql(pdb_file)

    #there is chain M
    if 'M' in sql.get_chains():
        checks['pdb_M'] = True
    else:
        checks['pdb_M'] = False
        return (ID, checks)

    #chain M has 170<x<184 res (store residues)
    M_length = len(set(sql.get('resSeq', chainID='M')))
    if 170<=M_length<=184:
        checks['M_len'] = M_length
    else:
        checks['M_len'] = False
        return (ID, checks)

    #there is chain P
    if 'P' in sql.get_chains():
        checks['pdb_P'] = True
    else:
        checks['pdb_P'] = False
        return (ID, checks)
    #chain P has 7<x<25 res (store residues)
    P_length = len(set(sql.get('resSeq', chainID='P')))
    if 7<=P_length<=25:
        checks['P_len'] = P_length
    else:
        checks['P_len'] = False
        return (ID, checks)

    #there is NO chain N
    if 'N' not in sql.get_chains():
        checks['no_N'] = True
    else:
        checks['no_N'] = False
        return (ID, checks)

    # M.pssm is in there
    M_pssm_file = f'{path}/pssm/{ID}.M.pdb.pssm'
    if os.path.isfile(M_pssm_file):
        checks['M_pssm']=True
    else:
        checks['M_pssm'] = False
        return (ID, checks)

    #has same n residues and numbering as pdb
    with open(M_pssm_file) as Mpssm:
        next(Mpssm)
        resIDs = [int(line.split()[0]) for line in Mpssm]
    if sorted(resIDs) == sorted(list(set(sql.get('resSeq', chainID='M')))):
        checks['M_pssm_as_pdb'] = True
    else:
        checks['M_pssm_as_pdb'] = False
        return (ID, checks)
        
    # P.pssm is in there
    P_pssm_file = f'{path}/pssm/{ID}.P.pdb.pssm'
    if os.path.isfile(P_pssm_file):
        checks['P_pssm']=True
    else:
        checks['P_pssm'] = False
        return (ID, checks)

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
        return (ID, checks)

    #ID, P chain lenght and seq correspond to csv data
    pdb_P_seq = sorted(list(set([tuple(x) for x in sql.get('resSeq,resName', chainID='P')])))
    pdb_P_seq = ('').join([aminoacids[aa[1]] for aa in pdb_P_seq])
    if pdb_P_seq == pssm_P_seq == df[df['ID']==ID]['peptide'].item():
        checks['P_chain_all_same'] = True
    else:
        checks['P_chain_all_same'] = False
        return (ID, checks)


    return (ID, checks)





db2_path = '/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/*/*'
paths = glob.glob(db2_path)

IDs_csv = '/projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv'
df = pd.read_csv(IDs_csv)

checks = Parallel(n_jobs = 128, verbose = 1)(delayed(check_case)(path, df) for path in paths)
checks = {x[0] : x[1] for x in checks}

with open('./db2_checks.tsv', 'w') as outfile:
    header = ['ID'] + list(list(checks.values())[0].keys())
    outfile.write(('\t').join(header) + '\n')
    for key in checks:
        row = [key] + [str(x) for x in list(checks[key].values())]
        outfile.write(('\t').join(row) + '\n')
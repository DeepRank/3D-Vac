import os
import pssmgen
from pdb2sql import pdb2sql
from glob import glob
import time
import argparse

arg_parser = argparse.ArgumentParser(
    description="""
    Merge PDB chains of class II models into one chain with ID: M
    """
)
arg_parser.add_argument("--input-folders", "-i",
    help="Path to the folders containing the pdbs to be merged",
    default="/projects/0/einf2380/data/pMHCII/pssm_mapped/BA/*/*",
)

a = arg_parser.parse_args()

def parse_pssm(pssm_file):
    pssm = []
    with open(pssm_file, 'r') as infile:
        for line in enumerate(infile):
            try:
                pssm.append(line)
            #print(line.split()[1])
            except:
                pass
    header = pssm[:1]
    #tail = pssm[-5:]
    #pssm = pssm[1:-5]
    pssm = pssm[1:]
    return header, pssm#, tail

infolders = glob(a.input_folders)
for infolder in infolders:
    t0 = time.time()
    print('Working on: ', infolder)
    #1 Parse pdb
    case = ('_').join(infolder.split('/')[-1].split('_')[:-1])
    #Copy pdb to not lose it
    pdbfile = infolder + f'/pdb/{case}.pdb'
    os.system(f'cp {pdbfile} {pdbfile}.origin')
    db = pdb2sql(pdbfile)
    if db.get_chains() != ['M', 'N', 'P']:
        print('Some chain is missing!')
        continue

    #2 Parse pssms
    M_pssm_file = infolder + f'/pssm/{case}.M.pdb.pssm'
    N_pssm_file = infolder + f'/pssm/{case}.N.pdb.pssm'

    #Copy M pssm to not lose it
    os.system(f'cp {M_pssm_file} {M_pssm_file}.origin')

    M_head, M_pssm = parse_pssm(M_pssm_file)
    N_head, N_pssm = parse_pssm(N_pssm_file)

    #Move the N pssm
    os.system(f'mv {N_pssm_file} {N_pssm_file}.origin')

    if len(M_pssm) > 90:
        print('M chain PSSM is too long!')
        continue

    #3 check pssm numbers and pdb numbers check out
    if (int(M_pssm[0][1].split()[0]),
    int(M_pssm[-1][1].split()[0]),
    int(N_pssm[0][1].split()[0]),
    int(N_pssm[-1][1].split()[0])) == (db.get('resSeq', chainID='M')[0],
    db.get('resSeq', chainID='M')[-1],
    db.get('resSeq', chainID='N')[0],
    db.get('resSeq', chainID='N')[-1]):
        print(f'PSSM and PDB numbers check out for case {case} !')
    else:
        raise Exception("Numbers don't check out!")

    #4 merge PDB chains. Maintain seq IDs
    #last_M = db.get('resSeq', chainID='M')[-1]
    index_modifier = 1000
    index_to_update= db.get('rowID', chainID='N')
    new_indices = list(map(lambda x:x+index_modifier, db.get('resSeq', chainID='N')))
    db.update_column('resSeq', values = new_indices, index=index_to_update)
    db.update_column('chainID', values = ['M' for x in range(len(index_to_update))], index=index_to_update)
    # Overrite pdb file
    db.exportpdb(pdbfile)

    #5 merge pssms in one
    new_N_pssm = []
    for line in N_pssm:
        index = line[0] + index_modifier
        row = line[1].split()
        row[0] = str(int(row[0]) + index_modifier)
        row[2] = str(int(row[2]) + index_modifier)
        tmp1 = ["{:>7s}".format(j) for j in row[:4]]
        tmp2 = ["{:>4s}".format(j) for j in row[4:]]
        new_N_pssm.append((index, " ".join(tmp1+tmp2) + "\n"))

    new_pssm = M_head + M_pssm + new_N_pssm
    
    # Write mrged pssm by overriding the old one
    with open(M_pssm_file, 'w') as outfile:
        for line in new_pssm:
            outfile.write(line[1])

    t1 = time.time()
    tf = t1 - t0
    print('Time: ', tf)
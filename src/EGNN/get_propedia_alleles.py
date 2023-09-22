import torch
import requests
import PANDORA.Pandora.Modelling_functions as mf
import pickle
from joblib import Parallel, delayed
from operator import xor
#Parallel(n_jobs = num_cores, verbose = 1)(delayed(run_case)(target) for target in list(self.targets.values()))

def get_pdb_allele(pdb):
    pdb_id = pdb.id.split('_')[0]
    mhc_id = pdb.id.split('_')[-1]
    pept_id = pdb.ligand_id.split('_')[0]
    if pdb.id.split('_')[1] != pdb.ligand_id.split('_')[0]:
        print(f'WARNING: Pept ID {pept_id} and case id {pdb.id} do not match')

    url =f'https://www.rcsb.org/fasta/entry/{pdb_id}/display'
    fasta = requests.get(url).text
    fasta_dict = {}
    #Get IDs per each chain
    for x in fasta.split('>'):
        if x != '':
            #Clear the ID field from extra characters and original chain ID
            #IDs = x.split('|')[1].replace('Chains', '').replace('Chain','').replace('auth','').replace('(','').replace(')','').replace(',','').split(' ') 
            string = x.split('|')[1].split('auth')[0]
            for ch in ['Chains', 'Chain', '(', ')' ,',', '[' , ']']:
                string = string.replace(ch,'')
            IDs = string.split(' ')
            for ID in IDs:
                if ID != '':
                    fasta_dict[ID]= x.split('\n')[1]
                    
    #Check that at least one of the two chains of the interaction is the peptide
    if mhc_id in fasta_dict.keys() and pept_id in fasta_dict.keys():
        #if xor(7 =< len(fasta_dict[mhc_id]) =< 15, 7 =< len(fasta_dict[pept_id]) =< 15):
        if 7 <= len(fasta_dict[pept_id]) <= 15:
            pass
        
        else:
            return None
    else:
        return None
    
    try:
        blast_results = mf.blast_mhc_seq(fasta_dict[mhc_id])
        max_value = max([x[1] for x in blast_results])
        best_match = [x for x in blast_results if x[1] == max_value]
        
    except Exception:
        return None
        '''
        print(f'Case {pdb_id} MHC chain ID is wrong. Trying to find it by blasting...')
        blast_scores = {}
        for ID in fasta_dict:
            try:
                blast_scores[ID] = mf.blast_mhc_seq(fasta_dict[ID])[0]
            except Exception:
                blast_scores[ID] = ('NA',0.0)
        blast_scores = sorted(blast_scores.items(), key=lambda x:x[1][1], reverse=True)
        best_match = blast_scores[0][1][0]
        print(f'Reported ID: {mhc_id}, Best BLAST match: {blast_scores[0][0]}')
        '''
    
    fasta_dict = {k : fasta_dict[k] for k in fasta_dict if k in [mhc_id, pept_id]}
    fasta_dict = {k[0] : k[1] for k in sorted(fasta_dict.items(), key=lambda x:len(x[1]), reverse=False)}
    fasta_dict['allele'] = best_match
    return (pdb_id, fasta_dict)


db = torch.load('propedia_residue_mhc_1.pt')

all_pdb = Parallel(n_jobs = 128, verbose = 1)(delayed(get_pdb_allele)(pdb) for pdb in db)

all_pdb_dict = {x[0] : x[1] for x in all_pdb if x != None}
    
    # if idx % 100 == 0 and idx > 0:
    #     print('Processed {}/{} files'.format(idx, len(db)))
            
with open('propedia_alleles.pkl', 'wb') as outpkl:
    pickle.dump(all_pdb_dict, outpkl)
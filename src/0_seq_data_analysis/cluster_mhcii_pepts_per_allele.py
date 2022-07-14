# -*- coding: utf-8 -*-

import pickle
from statistics import mean, stdev
import sys

sys.path.append('./')
from cluster_peptides import *

#%%
with open('../binding_data/pMHCII/ba_values.pkl', 'rb') as inpkl:
    ba = pickle.load(inpkl)
    ba_dict = pickle.load(inpkl)
    ba_c = pickle.load(inpkl)
    
    
ba_all_dict = {}
for x in ba:
    #if 'HLA-' in x[1] and '/' not in x[1]:67
    pept = x[0]
    allele = x[1]
    score = x[2]
    try:
        ba_all_dict[allele][pept].append(score)
    except KeyError:
        try:
            ba_all_dict[allele][pept] = [score]
        except KeyError:
            ba_all_dict[allele] = {pept:[score]}
        
#%%
matrix = 'PAM250'
n_jobs = int(sys.argv[1])
pept_len = 15
distances = {}
for allele in list(ba_all_dict.keys()):
    print('######')
    print('Working on: ',allele)
    peptides = [x for x in list(ba_all_dict[allele].keys()) if len(x) == pept_len]
    if len(peptides) > 1:
        scores = get_score_matrix(peptides, n_jobs, matrix)
        distances[allele] = scores
    else:
        print('Not enough %i-mer peptides for allele %s' %(pept_len, allele))
    
with open('./distances.pkl', 'wb') as outpkl:
    pickle.dump(distances, outpkl)
    
avg_scores = {x : (mean(distances[x]), stdev(distances[x])) for x in distances if x != [] and len(distances[x]) > 1}
with open('./distances.pkl', 'wb') as outpkl:
    pickle.dump(distances, outpkl)
    pickle.dump(avg_scores, outpkl)
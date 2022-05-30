#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:29:05 2021

@author: Dario Marzella
"""

import os
import pickle
from matplotlib import pyplot as plt
from math import exp, log
from joblib import Parallel, delayed
from collections import Counter

#%%
run_all = False

    
if run_all:
    #%%
    
    
    cell_to_alleles = {}
    with open('../binding_data/pMHCII/allelelist.txt') as infile:
        for line in infile:
            cell_to_alleles[line.split(' ')[0]] = line.split(' ')[1].replace('\n', '').split(',')
    
    acceptable_ids = {x:cell_to_alleles[x] for x in cell_to_alleles if len(cell_to_alleles[x]) == 1}
    
    global ba
    global el
    
    #%%
    #NetMHCPan rescaling function: 1-log(aff)/log(50,000)
    # aff = exp((NMP2_score) * log(50000))
    
    ##Get NetMHCPanI4.1 (NMP2) training data
    NMP2_data_folder = '../binding_data/pMHCII/'
    ba = []
    for f in os.listdir(NMP2_data_folder):
        if not f.startswith('allelelist') and 'BA' in f and f.endswith('.txt'):
            print(f)
            with open(NMP2_data_folder + f, 'r') as infile:
                for line in infile:
                    row = line.split('\t')
                    affinity = exp((float(row[1])) * log(50000))
                    if row[2] in list(acceptable_ids.keys()):
                        ba.append([row[0], row[2].replace('\n', ''), float(row[1]), affinity]) 
    
    #%% Remove ba redundancies
    ba = set(tuple(x) for x in ba)
    
    #%% Make NMP2 dictionary
    ba_dict = {}
    for x in ba:
        #if 'HLA-' in x[1] and '/' not in x[1]:67
        try:
            ba_dict[x[0]][x[1]].append(x[2])
        except KeyError:
            ba_dict[x[0]] = {x[1]:[x[2]]}
                
    #%%
    ##Get NetMHCPanI4.1 (NMP2) training data
    el = []
    for f in os.listdir(NMP2_data_folder):
        if not f.startswith('allelelist') and 'EL' in f and f.endswith('.txt'):
            print(f)
            with open(NMP2_data_folder + f, 'r') as infile:
                for line in infile:
                    row = line.split('\t')
                    if row[2] in list(acceptable_ids.keys()):
                        el.append([row[0], row[2].replace('\n', ''), row[1]]) 
    
    #%% Remove el redundancies
    el = set(tuple(x) for x in el)
    
    #%% Make NMP2 dictionary
    el_dict = {}
    for x in el:
        #if 'HLA-' in x[1] and '/' not in x[1]:
        try:
            el_dict[x[0]][x[1]].append(x[2])
        except KeyError:
            el_dict[x[0]] = {x[1]:[x[2]]}
          
            
    #%% Do allele counters
    
    ba_c = Counter([x[1] for x in ba])
    ba_c = {k: v for k, v in sorted(ba_c.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True)}
    
    el_c = Counter([x[1] for x in el])
    el_c = {k: v for k, v in sorted(el_c.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True)}
#%%Pickle
with open('../binding_data/pMHCII/ba_values.pkl', 'wb') as outpkl:
    pickle.dump(ba,outpkl)
    pickle.dump(ba_dict,outpkl)
    pickle.dump(ba_c,outpkl)
    
with open('../binding_data/pMHCII/el_values.pkl', 'wb') as outpkl:
    pickle.dump(el,outpkl)
    pickle.dump(el_dict,outpkl)
    pickle.dump(el_c,outpkl)

#%% Unpickle

with open('../binding_data/pMHCII/ba_values.pkl', 'rb') as inpkl:
    ba = pickle.load(inpkl)
    ba_dict = pickle.load(inpkl)
    ba_c = pickle.load(inpkl)
    
with open('../binding_data/pMHCII/el_values.pkl', 'rb') as inpkl:
    el = pickle.load(inpkl)
    el_dict = pickle.load(inpkl)
    el_c = pickle.load(inpkl)

#%% Plot allele counters
X = [i*2 for i, x in enumerate(ba_c.keys())]
plt.scatter(X, ba_c.values() )
plt.yticks(range(0, 10000, 1000))
plt.xticks(ticks=range(0, len(ba_c.keys())*2,2),
           labels=ba_c.keys(),
           rotation=90, fontsize=4)
plt.show()
plt.clf()

#%% Preapare data for pie chart

ba_pos = []
ba_neg = []
for row in ba:
    if row[3] <= 500:
        ba_pos.append(row)
    else:
        ba_neg.append(row)
        
el_pos = []
el_neg = []
for row in el:
    if row[2] == '1':
        el_pos.append(row)
    else:
        el_neg.append(row)

tot_len = len(ba) + len(el)  

#%% Plot piechart of EL vs BA, pos vs neg

plt.pie([len(el)/tot_len*100,len(ba)/tot_len*100], labels=['EL: %i '%len(el), 'BA %i'%len(ba)], autopct='%1.1f%%')
plt.title('Total: ' + str(tot_len) + ' Experimental Values') 
plt.show()
plt.clf()

plt.pie([len(el_pos), len(el_neg), len(ba_pos), len(ba_neg)], 
        labels = ['EL pos: %i' %len(el_pos), 'EL neg: %i' %len(el_neg), 'BA pos: %i' %len(ba_pos), 'BA neg: %i' %len(ba_neg)], autopct='%1.1f%%')
plt.title('Total: ' + str(tot_len) + ' Experimental Values') 
plt.savefig('../plots/pMHCII/data_analysis/netmhciipan4.0_ba_el_posneg_piechart.png')
plt.show()
plt.clf()

#%% Plot peptides lengths

#Plot peptides lengths for ba
ba_len_counter = Counter([len(x[0]) for x in ba])

plt.scatter(ba_len_counter.keys(), ba_len_counter.values())
plt.title('Binding Affinity peptides length')
plt.xlabel('Peptide length')
plt.ylabel('N of peptides')
plt.savefig('../plots/pMHCII/data_analysis/ba_pept_len.png')
plt.show()
plt.clf()


#Plot peptides lengths for el
el_len_counter = Counter([len(x[0]) for x in el])

plt.scatter(el_len_counter.keys(), el_len_counter.values())
plt.title('Elution peptides length')
plt.xlabel('Peptide length')
plt.ylabel('N of peptides')
plt.savefig('../plots/pMHCII/data_analysis/el_pept_len.png')
plt.show()
plt.clf()

#Plot peptides lengths for el
tot_len_counter = Counter([len(x[0]) for x in el + ba])

plt.scatter(tot_len_counter.keys(), tot_len_counter.values())
plt.title('All peptides length')
plt.xlabel('Peptide length')
plt.ylabel('N of peptides')
plt.savefig('../plots/pMHCII/data_analysis/all_pept_len.png')
plt.show()
plt.clf()

'''
#%%

## Find redundant data
   
def find_overlaps(cases):
    #pept, allele, label
    overlaps = []
    for case in cases:
        pept=case[0]
        allele = case[1]
        label = case[2]
        matches = []
        if label == 'BA':
            for d in ba:
                if d[0] == pept and d[1] == allele:
                    matches.append(d)
        elif label == 'EL':
            for d in el:
                if d[0] == pept and d[1] == allele:
                    matches.append(d)
        overlaps.append(matches)
    return overlaps

#TODO: change these lines to give a batch of cases to find_overlaps
redundancies = Parallel(n_jobs=20)(delayed(find_overlaps)((x[0], x[1], 'BA')) for x in ba)
redundancies = filter([], redundancies)

print('################')
print('BA REDUNDANCIES')
print(redundancies)
print('################')
print('\n')

redundancies = Parallel(n_jobs=20)(delayed(find_overlaps)(x[0], x[1], 'EL') for x in el)
redundancies = filter([], redundancies)

print('################')
print('EL REDUNDANCIES')
print(redundancies)
'''
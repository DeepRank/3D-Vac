#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:15:45 2021

@author: Dario Marzella
"""

import os
#%% Get data


# Get Target names, alleles and binding
all_cases = {}
pos_neg = {'Positive' : 1,
           'Negative' : 0}

with open('/home/dariom/3d-epipred/binding_data/IDs_qual_human_complete.csv') as cases_infile:
    for line in cases_infile:
        row = line.replace('\n','').split(',')
        name = row[0]
        allele = row[1]
        binding = pos_neg[row[4]]
        
        all_cases[name] = {'allele': allele, 
                           'binding': binding}
        
# Get modelled cases
already_modelled = []
mod_dir = '/mnt/csb/Dario/3d_epipred/models'
for folder in os.listdir(mod_dir):
    case = folder.split('_')[0]
    model = case + '.BL00190001.pdb'
    scores = 'molpdf_DOPE.tsv'
    case_folder = mod_dir + '/%s/' %folder
    if model in os.listdir(case_folder) and scores in os.listdir(case_folder):
        already_modelled.append(case)
        all_cases[case]['folder'] = folder


#%%
# Keep only modelled targets (map)
#def return_unmodelled(case):
#    if len(case) == 2:
#        return case
    #if case not in already_modelled:
    #    return case

all_cases = {case : all_cases[case] for case in all_cases if len(all_cases[case]) == 3}

# Get best molpdf model for each target and soft link it to folder
for case in all_cases:
    all_cases[case]['best_model'] = os.popen('head -n 1 %s/%s/molpdf_DOPE.tsv' %(mod_dir, all_cases[case]['folder'])).read().split('\t')[0]
    if all_cases[case]['allele'].startswith('HLA-A'):
        os.popen('ln -s %s/%s/%s /home/dariom/3d-epipred/CNN/pdb/train/' %(mod_dir, all_cases[case]['folder'], all_cases[case]['best_model'])).read()
    elif all_cases[case]['allele'].startswith('HLA-B'):
        os.popen('ln -s %s/%s/%s /home/dariom/3d-epipred/CNN/pdb/test/' %(mod_dir, all_cases[case]['folder'], all_cases[case]['best_model'])).read()

#%% 
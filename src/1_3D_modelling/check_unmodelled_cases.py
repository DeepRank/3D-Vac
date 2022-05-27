import os

## Usage: python get_unmodelled_cases.py

#1. Open cases file and read the number of cases.
all_cases = {}
#with open('/home/dariom/3d-epipred/binding_data/IDs_qual_human_complete.csv') as cases_infile:
with open('../../binding_data/IDs_qual_human_complete.csv', 'r') as cases_infile:
    for line in cases_infile:
        row = line.replace('\n','').split(',')
        name = row[0]
        columns = ['allele', 'peptide', 'affinity', 'bind']
        all_cases[name] = {x : y for x,y in zip(columns,row[1:5])}
print('ALL CASES LENGHT: %i' %len(all_cases))

#%%
to_model = {}
#with open('/home/dariom/3d-epipred/binding_data/unmodelled_IDs_qual_human_complete.csv', 'r') as tomodel_infile:
with open('../../binding_data/unmodelled_IDs_qual_human_complete.csv', 'r') as tomodel_infile:
    for line in tomodel_infile:
        row = line.replace('\n','').split(',')
        name = row[0]
        columns = ['allele', 'peptide', 'affinity', 'bind']
        to_model[name] = {x : y for x,y in zip(columns,row[1:5])}
print('TO MODEL: %i' %len(to_model))
    

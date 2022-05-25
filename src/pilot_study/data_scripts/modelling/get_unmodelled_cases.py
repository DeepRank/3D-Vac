import os

## Usage: python get_unmodelled_cases.py

#TODO: 
#1. Open cases file and read the number of cases.
all_cases = {}
with open('/home/lepikhovd/binding_data/BA_pMHCI.csv') as cases_infile:
    for line in cases_infile:
        row = line.replace('\n','').split(",")
        name = row[0]
        all_cases[name] = line
print('ALL CASES LENGHT: %i' %len(all_cases))

#2. Open output folder. Check how many cases have model number 20. 
already_modelled = []
model_dir = '/projects/0/einf2380/data/pMHCI/models/temp';
for folder in os.listdir(model_dir):
    case = "_".join(folder.split('_')[0:2])
    model = case + '.BL00200001.pdb'
    scores = 'molpdf_DOPE.tsv'
    case_folder = f"{model_dir}/{folder}"
    if model in os.listdir(case_folder) and scores in os.listdir(case_folder):
        already_modelled.append(case)
        del(all_cases[case])

print('Already modelled LENGHT: %i' %len(already_modelled))
print('ALL CASES LENGHT after: %i' %len(all_cases))

#3. Write new input file without cases already modelled.
with open('/home/lepikhovd/binding_data/to_model.csv', 'w') as outfile:
    for key in all_cases:
        outfile.write(all_cases[key])


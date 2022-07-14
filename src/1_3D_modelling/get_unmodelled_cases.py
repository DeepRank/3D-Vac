import os
import sys
import glob

csv_path = sys.argv[1]

## Usage: python get_unmodelled_cases.py

#TODO: 
#1. Open cases file and read the number of cases.
all_cases = {}
with open(csv_path) as cases_infile:
    for line in cases_infile:
        row = line.replace('\n','').split(",")
        name = row[0]
        all_cases[name] = line
print('ALL CASES LENGHT: %i' %len(all_cases))

#2. Open output folder. Check how many cases have model number 20. 
already_modelled = []
model_dirs = glob.glob('/projects/0/einf2380/data/pMHCI/models/BA/*/*');
for folder in model_dirs:
    n_structures = len(glob.glob(f"{folder}/*.BL*.pdb"))
    if n_structures == 20:
        case = "_".join(folder.split("/")[-1].split("_")[0:2])
        already_modelled.append(case)
        del(all_cases[case])

print('Already modelled LENGHT: %i' %len(already_modelled))
print('ALL CASES LENGHT after: %i' %len(all_cases))

# #3. Write new input file without cases already modelled.
# with open('/home/lepikhovd/binding_data/to_model.csv', 'w') as outfile:
#     for key in all_cases:
#         outfile.write(all_cases[key])


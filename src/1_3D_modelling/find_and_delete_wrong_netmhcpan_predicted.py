import os
import sys
import pickle
sys.path.append('/home/dariom/PANDORA_new_test/')

from PANDORA.Pandora import Modelling_functions as mf

from joblib import Parallel, delayed
## Usage: python get_unmodelled_cases.py
# last submission ID: 1039787
#TODO: 
#1. Open cases file and read the number of cases.
all_cases = {}
with open('/home/dariom/3d-epipred/binding_data/IDs_qual_human_complete.csv') as cases_infile:
    for line in cases_infile:
        row = line.replace('\n','').split(',')
        name = row[0]
        all_cases[name] = row[1:]
print('ALL CASES LENGHT: %i' %len(all_cases))


n_jobs = int(sys.argv[1])
#2. Open output folder. Check how many cases have model number 20. 
#for folder in os.listdir('/mnt/csb/Dario/3d_epipred/models'):
def get_wrong_models(batch_dim, batch_id):
    
    start = batch_id * batch_dim
    if batch_id == n_jobs -1:
        end = None
    else:
        end = (batch_id +1) * batch_dim
        
    wrong_cases = []
    
    for folder in cases[start:end]:
        case = folder.split('_')[0]
        model = case + '.BL00200001.pdb'
        scores = 'molpdf_DOPE.tsv'
        case_folder = '/mnt/csb/Dario/3d_epipred/models/%s/' %folder
        if model in os.listdir(case_folder) and scores in os.listdir(case_folder):
            #already_modelled[case] = all_cases[case]
            peptide = all_cases[case][1]
            allele = all_cases[case][0]
            anchors = mf.predict_anchors_netMHCpan(peptide,
                                          [allele], 
                                          verbose=False, rm_output=True)
            
            with open(case_folder + 'MyLoop.py') as myloop:
                for line in myloop:
                    if 'return M.selection(self.residue_range(' in line:
                        
                        modelled_anchs = ['na', 'na']
                        modelled_anchs[0] = int(line.split("'")[1].split(":")[0])
                        if modelled_anchs[0] > 1:
                            modelled_anchs[0] -= 1
                            
                        modelled_anchs[1] = int(line.split("'")[3].split(":")[0])
                        if modelled_anchs[1] < len(peptide):
                            modelled_anchs[1] += 1
            
            if anchors != modelled_anchs:
                print("Anchors don't match for case " + folder)
                print("Anchors:   ", anchors)
                print("Modelled:  ", modelled_anchs)
                print("Allele: ", allele)
                
                wrong_cases.append([folder, (';').join(str(x) for x in anchors), (';').join(str(x) for x in modelled_anchs), peptide, allele])
            else:
                print('Case %s OK' %case)
        else:
            pass
            #raise Exception('STOP')
    
    return wrong_cases
        #del(all_cases[case])

global cases
cases_dir = '/mnt/csb/Dario/3d_epipred/models'
cases = os.listdir(cases_dir)

to_print = Parallel(n_jobs = n_jobs, backend='multiprocessing')(delayed(get_wrong_models)(int(len(cases)/n_jobs), i) for i, folder in enumerate(cases))
#to_print = [x for x in to_print if x != None]
to_print = sum(to_print, [])

with open('/home/dariom/3d-epipred/modelling/bugged_anchors.pkl', 'wb') as outpkl:
    pickle.dump(to_print, outpkl)
    
with open('/home/dariom/3d-epipred/modelling/bugged_anchors.tsv', 'w') as outfile:
    header = ['Folder', 'Correct prediction', 'Bugged prediction', 'peptide', 'allele']
    outfile.write(('\t').join(header) + '\n')
    for case in to_print:
        outfile.write(('\t').join(case) + '\n')
# print('Already modelled LENGHT: %i' %len(already_modelled))
# print('ALL CASES LENGHT after: %i' %len(all_cases))


# #3. Delete wrong cases
for case in to_print:
    os.popen('rm -r %s/%s' %(cases_dir, case[0])).read()


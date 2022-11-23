# this script allows to copy a small fraction of the real data on a user's folder for fast experimenting with the code
import glob
import shutil
import os


src_dir = "/projects/0/einf2380/data/pMHCI/features_input_folder/exp_nmers_all_HLA_quantitative"
dest_dir = "/home/ccrocion/snellius_data_sample/data/pMHCI/features_input_folder/exp_nmers_all_HLA_quantitative"
n_files = 100

pdb_files = glob.glob(os.path.join(src_dir + '/pdb', '*.pdb'))
count = 0
for pdb_file in pdb_files:
    pdb_id = pdb_file.split('/')[-1].split('.')[0]
    pssm_files = glob.glob(os.path.join(src_dir + '/pssm', pdb_id + '.*.pssm'))
    # not all peptides' pssms are present, selecting only the complete data points
    if len(pssm_files) == 2:
        shutil.copy(pdb_file, dest_dir + '/pdb')
        print(f'copied {pdb_file}')
        for pssm_file in pssm_files:
            shutil.copy(pssm_file, dest_dir + '/pssm')
            print(f'copied {pssm_file}')
        print('\n')
        count += 1
        if count == n_files:
            break

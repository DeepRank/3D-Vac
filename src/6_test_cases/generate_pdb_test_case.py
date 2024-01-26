from PANDORA import Database
from PANDORA.Wrapper import Wrapper
import glob
import shutil


in_csv = '/home/dmarz/3D-Vac/src/5_test_cases/to_model.csv'
work_folder = '/home/dmarz/test_cases/work_folders/'
wrapper_id = 'NOnetmhcpan'
final_folder = f'/home/dmarz/test_cases/final_folders/{wrapper_id}'
num_cores = 64

db = Database.load()

wrap = Wrapper.Wrapper(in_csv, db, MHC_class='I', 
                    IDs_col=0,peptides_col=1, allele_name_col=2,
                    collective_output_dir=work_folder, archive=False,
                    benchmark=False, verbose=True, delimiter=',',
                    header=True, num_cores=num_cores, use_netmhcpan=False,
                    n_loop_models=20, clip_C_domain=True,
                    restraints_stdev=0.3,wrapper_id=wrapper_id)



for folder in glob.glob(f'{work_folder}{wrapper_id}/*'):
    try:
        with open(f'{folder}/molpdf_DOPE.tsv', 'r') as scorefile:
            best_model = [line.replace('\n','').split('\t')[0] for line in scorefile][0]
            
        shutil.copyfile(f'{folder}/{best_model}', f'{final_folder}/{best_model}')
    except FileNotFoundError:
        print(f'FileNotFoundError for case {folder}')
        continue
    
    


import os
import sys
import re
import glob
import argparse
import traceback
import subprocess
import tarfile
from joblib import Parallel, delayed 
import pandas as pd

arg_parser = argparse.ArgumentParser(description="Script used to generate a list of unmodelled p:MHC complexes by comparing \
the output folder and the initial db1. A case is considered modelled if 19 pdb structures were generated. Here 19 is the \
threshold because modeller sometimes is able to generate only 19/20 in very extreme cases (6 out of 11K cases). Therefore, \
even extreme cases are considered modelled. The logs on how much cases are still to model is available at the printed logs \
which is at project_folder/data/modelling_logs/clean_models_job.log")

arg_parser.add_argument("--csv-file", "-f",
    help="Path to db1 containing the p:MHC complexes to model.",
    default="../../data/external/processed/I/BA_pMHCI.csv",
)
arg_parser.add_argument("--to-model", "-t",
    help="Path to to_model.csv file ",
    default="../../data/external/processed/I/to_model.csv",
)
arg_parser.add_argument("--update-csv", "-u",
    help="If this argument is provided, the `to_model.csv` file is updated with the unmodelled cases and the number \
    of cases to model is printed into the log file. If not provided, it prints the number of cases to model without \
    updating the `to_model.csv`",
    default=False,
    action="store_true"
)
arg_parser.add_argument("--models-dir", "-m",
    help="Path to the BA or EL folder where the models are generated",
    default="/projects/0/einf2380/data/pMHCI/models/BA",
)
arg_parser.add_argument("--parallel", "-p",
    action='store_true',
    help="Run script in parallel on a slurm cluster with multiple nodes",
)
arg_parser.add_argument("--archived", "-a",
    action='store_true',
    help="Flag to be used when folders are archived",
)

a = arg_parser.parse_args()


def zip_and_remove(case):    
    # remove the old archive, but only if already unzipped
    if os.path.exists(f'{case}.tar') and os.path.exists(case):
        subprocess.check_call(f"rm {case}.tar", shell=True)   
    # create new archive of the folder
    with tarfile.open(f'{case}.tar', 'w') as archive: # create new tarfile to gather files in
        case_files = glob.glob(os.path.join(case, '*'))
        for case_file in case_files:
            archive.add(case_file)
    # check if tar was created correctly and remove the original files from the folder
    if os.path.exists(f'{case}.tar'):
        subprocess.check_call(f"rm -r {case}", shell=True)
        return True
    else:
        print(f'Error creating archive: {case}.tar, skipping the file removal')
        return False

def get_archive_members(dir):
    try:
        with tarfile.open(f'{dir}.tar', 'r') as archive:
            members = [mem.name for mem in archive]
            return members
    except tarfile.ReadError as e:
        print(e)
        print(f'Removing empty or corrupt tar file: {dir}.tar')
        subprocess.Popen(f'rm -r {dir}.tar', shell=True).wait()
        return
    except FileNotFoundError:
        if os.path.exists(dir):
            print(f"No tar found but directory exists: {dir}\n trying to archive again")
            if zip_and_remove(dir):
                return get_archive_members(dir)
    except:
        print(traceback.print_exc())


#1. Open cases file
df = pd.read_csv(f"{a.csv_file}")
all_cases = len(df)

wildcard_path = os.path.join(a.models_dir, '*/*')
folders = glob.glob(wildcard_path)
model_dirs = [dir.split('.')[0] for dir in folders]
n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))
# if a.parallel:
#     node_index = int(os.getenv('SLURM_NODEID'))
#     n_nodes = int(os.getenv('SLURM_JOB_NUM_NODES'))


#     step = math.ceil(len(folders)/n_nodes)
#     start = node_index*step
#     end = start + step
#     if end > len(folders):
#         end = len(folders)

#     print(f'Start: {start} \nEnd {end} \nLen folders {len(folders)}')

#     if node_index != n_nodes-1:
#         model_dirs = folders[start:end]
#     else:
#         model_dirs = folders[start:]
# else:
#      model_dirs = glob.glob(os.path.join(a.models_dir, '*/*'))


def search_folders(folder):
    try:
        #2. Open output folder. If the folder doesn't have 20 pdb files, it is considered as unmodelled.     #
        if a.archived:
            members = get_archive_members(folder)
            if not members:
                return
            search_pdb = [re.search(r'BL.*\.pdb$', member) for member in members]
            n_structures = sum((i is not None for i in search_pdb))
        else:
            n_structures = len(glob.glob(f"{folder}/*.BL*.pdb"))
        if n_structures >= 19 and n_structures <= 20: # the n_structures <= 20 is to be sure that no more than 20 structures are
            # generated
            case = "_".join(folder.split("/")[-1].split("_")[0:2])
            return case
    except:
        print(f'Failed to search folder for case {folder}')
        print(traceback.format_exc())

if a.parallel:
    cases = Parallel(n_jobs = n_cores, verbose = 1)(delayed(search_folders)(case) for case in list(model_dirs))
else:
    cases = []
    for folder in folders:
        cases.append(search_folders(model_dirs))

# gather the indices of the IDs that were found to have complete models
cases_indices = df[df["ID"].isin(cases)].index.tolist()
df.drop(cases_indices, inplace=True) # the initial list of cases is reduced with the case.


print(f"Initial number of cases: {all_cases}")
print(f'Unmodelled: {len(df)}')

# #3. Write new input file without cases already modelled.
if a.update_csv:
    df.to_csv(a.to_model, index=False) # the initial list of cases without the modelled cases
# is returned

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
    default="/projects/0/einf2380/data/pMHCI/3D_models/BA/\*/\*",
)
arg_parser.add_argument("--parallel", "-p",
    action='store_true',
    help="Run script in parallel on a slurm cluster with multiple nodes",
)
arg_parser.add_argument("--archived", "-a",
    action='store_true',
    help="Flag to be used when folders are archived",
)

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

def check_molpdf(fold):
    try:
        with tarfile.open(f'{fold}.tar', 'r') as archive:
            molpdf = archive.extractfile(f'{fold[1:]}/molpdf_DOPE.tsv')
            molpdf_df = pd.read_csv(molpdf, sep='\t', header=None)
            if molpdf_df.shape[0] >= 19:
                df_types = []
                for idx, row in molpdf_df.iterrows():
                    df_types.append(pd.isna(row[1]))
                    df_types.append(pd.isna(row[2]))
                # allow 4 faulty scores in the molpdf file (2 models)
                if df_types.count(True) <= 4:
                    return True
    except tarfile.ReadError as e:
        print(e)
    except KeyError as e:
        print(f'Molpdf.tsv not present: {fold}')
    except pd.errors.EmptyDataError:
        print(f'Molpdf empty: {fold}')


def search_folders(folder):
    try:
        # Open output folder. If the folder doesn't have 20 pdb files, it is considered as unmodelled.     #
        if a.archived:
            members = get_archive_members(folder)
            if not members:
                return
            search_pdb = [re.search(r'BL.*\.pdb$', member) for member in members]
            n_structures = sum((i is not None for i in search_pdb))
        else:
            n_structures = len(glob.glob(f"{folder}/*.BL*.pdb"))
        if n_structures >= 19 and n_structures <= 20: # the n_structures <= 20 is to be sure that no more than 20 structures are
            case = "_".join(folder.split("/")[-1].split("_")[0:2])
            molpdf_present = [re.search('molpdf_DOPE.tsv', mem) for mem in members]
            
            # return True if 19 or 20 models present AND molpdf present AND check molpdf has valid values
            if molpdf_present:
                if check_molpdf(folder):
                    return case
        try:
            print(f'Removing case from models folder: {folder}')
            subprocess.Popen(f'rm -r {folder}.tar', shell=True).wait()
        except Exception as e:
            print(e)
            traceback.print_exc()
    except:
        print(f'Failed to search folder for case {folder}')
        print(traceback.format_exc())

a = arg_parser.parse_args()

#1. Open cases file
df = pd.read_csv(f"{a.csv_file}")
all_cases = len(df)

#2. Get all the paths of the modelled cases in the models folder
if "*" not in a.models_dir and type(a.models_dir)!=list:
    print("Expected a wild card path, please provide a path like this: mymodelsdir/\*/\*")
    raise SystemExit

wildcard_path = a.models_dir.replace('\\', '')
folders = glob.glob(wildcard_path)
model_dirs = [dir.split('.')[0] for dir in folders]

#3. Count successfully modelled cases based on some critera
if a.parallel:
    n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))
    cases = Parallel(n_jobs = n_cores, verbose = 1)(delayed(search_folders)(case) for case in list(model_dirs))
else:
    cases = []
    for folder in model_dirs:
        cases.append(search_folders(folder))

#4. gather the indices of the IDs that were found to have complete models
cases_indices = df[df["ID"].isin(cases)].index.tolist()
#5. Drop the cases from the original dataframe
df.drop(cases_indices, inplace=True) # the initial list of cases is reduced with the case.

print(f"Initial number of cases: {all_cases}")
print(f'Unmodelled: {len(df)}')

print('List of cases that were present in output folder but were not valid, these are now removed:')
unmodelled_cases = [False for case in cases if case==None]
for bcase, casedir in zip(cases, model_dirs):
    if not bcase:
        print(casedir)

# #3. Write new input file without cases already modelled.
if a.update_csv:
    df.to_csv(a.to_model, index=False) # the initial list of cases without the modelled cases
else:
    print('Unmodelled cases (excluding removed cases):')
    not_modelled_ids = df['ID'].tolist()
    for mod_id in not_modelled_ids:
        print(mod_id)

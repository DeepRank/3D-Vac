import os
import argparse
import glob
import math
import tarfile
import subprocess
import traceback
import pandas as pd
from joblib import Parallel, delayed 

arg_parser = argparse.ArgumentParser(
    description="""
    Script used to clean the models from unecessary files after `modelling_job.py` has completed.
    """
)
arg_parser.add_argument("--models-path", "-p",
    help = "glob.glob() string argument to generate a list of all models. A short tutorial on how to use glob.glob: \
    https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/\
     Default value: \
    /projects/0/einf2380/data/pMHCI/models/BA",
    default = "/projects/0/einf2380/data/pMHCI/models/BA"
)
arg_parser.add_argument("--mhc-class", "-m",
    help="MHC class of the cases",
    choices=['I','II'],
    required=True,
)
arg_parser.add_argument("--structure-rank", "-r",
    help="""
    How many pdb structure to pick from each case. This argument corresponds to the n-th rank. For instance if -r 4,
    the number of structures selected from each case will be the 4 first structures in the molpdf.tsv file. For now,
    This whole pipeline is working only for 1 structure per db2 case. Therefore this argument shouldn't be tweaked, for now.
    Default 1.""",
    default=1,
    type=int,
)
arg_parser.add_argument("--single-path", "-s",
    help="Give the path of a single case, for debugging",
    type=str,
    required=False)

a = arg_parser.parse_args()

db2_selected_models_path = f"/projects/0/einf2380/data/pMHC{a.mhc_class}/db2_selected_models_1"

if a.single_path:
    folders = [a.single_path]
    n_cores = 1
else:
    # clean the whole models folder
    wildcard_path = os.path.join(a.models_path, '*/*')
    folders = glob.glob(wildcard_path)
    folders = [dir.split('.')[0] for dir in folders]
    n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))

#######################################################

# check if case was modelled:
def check_exist(dir):
    # if both the folder and the tar exist, remove the folder and extract tar again (we expect the tar not to be touched)
    if os.path.exists(f'{dir}.tar') and os.path.exists(dir):
        subprocess.Popen(f'rm -r {dir}', shell=True).wait()
        return dir
    elif os.path.exists(dir):
        print('Only the folder of case exists, removing folder')
        subprocess.Popen(f'rm -r {dir}', shell=True).wait()
    elif os.path.exists(f'{dir}.tar'):
        return dir
    # elif os.path.exists(dir):
    #     # arhive again if folder was unzipped during process, will return true if successful
    #     return zip_and_remove(dir)
        # print(f"Case: {dir} was not modelled")

def unpack_archive(dir):
    try:
        with tarfile.open(f'{dir}.tar', 'r') as tar:
            # need to rename tar members because writing to the original dir will not work otherwise
            for member in tar:
                member.name = os.path.basename(member.name)
            os.mkdir(dir)
            tar.extractall(dir)
    except tarfile.ReadError as e:
        print(e)
        # remove complete directory so that get_unmodelled cases sees it as unmodelled
        print(f"(unpack archive) Tar file is not valid, removing: {dir}")
        subprocess.Popen(f'rm -r {dir}.tar', shell=True).wait()
        return False
    except:
        traceback.print_exc()
        return False
    assert os.path.exists(dir) # if all goes well the tar file should exist
    return True

# cleaning of target directories, to be called in parallel
def clean_target_dir(dir):
    for dir_wild in ['/*.DL*', '/*.B99990001.pdb', '/*.V99990001', '/*.D00000001', 
                    '/__pycache__/', '/*.lrsr', '/*.rsr', '/*.sch']:
        wild_paths = glob.glob(f'{dir}{dir_wild}')
        for file_path in wild_paths:
            if os.path.exists(file_path):
                if dir_wild[-1] == '/':
                    subprocess.Popen(f'rm -r {file_path}', shell=True).wait()
                else:
                    subprocess.Popen(f'rm {file_path}', shell=True).wait()


# zipping the folders to reduce the amount of files (inodes)
def copy_target_files(case):
    # get the best structure
    molpdf_path = f"{case}/molpdf_DOPE.tsv"
    if os.path.exists(molpdf_path):
        molpdf_df = pd.read_csv(molpdf_path, sep="\t", header=None)
        target_scores = molpdf_df.iloc[:,1].sort_values()[0:a.structure_rank]
        target_mask = [score in target_scores.tolist() for score in molpdf_df.iloc[:,1]]
        target_ids = molpdf_df[target_mask].iloc[:,0]
        structures = [f"{case}/{structure}" for structure in target_ids]

        # symlink each target
        # attempt to create subfolders:
        for structure in structures:
            dir = "/".join(structure.split("/")[-4:-1])
            pdb_file = structure.split("/")[-1].replace("_", "-")
            destination_dir = f"{db2_selected_models_path}/{dir}/pdb"
            destination_file = f"{destination_dir}/{pdb_file}"
            try: # create remaining subfolders:
                os.makedirs(destination_dir); 
            except FileExistsError:
                print('Destination file exists', destination_dir)
                if os.path.exists(destination_file):
                    print('best model alredy copied, next..')
                    continue
            except:
                print('Something went wrong in creating ', destination_dir)
                print(traceback.format_exc())
            try: #make the symlink:
                subprocess.check_call(f'cp {structure} {destination_file}', shell=True)
            except:
                print(traceback.format_exc())
    # elif os.path.exists(f"{db2_selected_models_path}/pdb/*"): # this path might not be right need to change
    #         print("Molpdf file was already moved: {case}")
    else:
        print(f'Molpdf path does not exist or was already moved: {case}')

def zip_and_remove(case):    
    # remove the old archive, but only if already unzipped
    if os.path.exists(f'{case}.tar') and os.path.exists(case):
        subprocess.Popen(f"rm {case}.tar", shell=True).wait()
    # create new archive of the folder
    with tarfile.open(f'{case}.tar', 'w') as archive: # create new tarfile to gather files in
        case_files = glob.glob(os.path.join(case, '*'))
        for case_file in case_files:
            archive.add(case_file)
    # check if tar was created correctly and remove the original files from the folder
    if os.path.exists(f'{case}.tar'):
        subprocess.Popen(f"rm -r {case}", shell=True).wait()
        return True
    else:
        print(f'Error creating archive: {case}.tar, skipping the file removal')
        return False

def clean_copy_target_file(dir):

    try:
        case_dir = check_exist(dir)
        # if case was modelled, do the next steps
        if case_dir:
            print('got through check')
            if not unpack_archive(dir): # skip case if files are not valid
                return 
            clean_target_dir(dir)
            copy_target_files(dir)       
            zip_and_remove(dir)
    except:
        print(f'Cleaning failed for case {dir}')
        print(traceback.format_exc())
    try:
        # try to archive folder and remove unarchived files after an exception has occured
        zip_and_remove(dir)
    except:
        print(traceback.format_exc())

step = math.ceil(len(folders)/10)

# print(f'Cut folders: start:{start} end:{end}')
# cut the process into pieces to make sure not too many files are unzipped at the same time
for chunk in range(0,10):
    start = chunk*step
    end = start+step
    if end > len(folders):
        end = len(folders)
    chunk_folders = folders[start:end]
    print(f'Chunk folders: start:{start} end:{end}')
    Parallel(n_jobs = n_cores, verbose = 1)(delayed(clean_copy_target_file)(case) for case in list(chunk_folders))

print(f"Cleaning finished.")

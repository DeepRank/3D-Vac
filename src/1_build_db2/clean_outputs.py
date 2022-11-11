import os
import re
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
arg_parser.add_argument("--models-dir", "-p",
    help = "glob.glob() string argument to generate a list of all models. A short tutorial on how to use glob.glob: \
    https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/\
     Default value: \
    /projects/0/einf2380/data/pMHCI/3D_models/BA/\*/\*",
    default = "/projects/0/einf2380/data/pMHCI/3D_models/BA/\*/\*"
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
arg_parser.add_argument("--sequential", "-s",
    help="Run the program sequentially instead of in parallel",
    action='store_true',
    default=False
)

# check if case was modelled:
def check_exist(case:str):
    """check if the case was modelled, also make sure that an open folder 
    and tar do not exist at the same time

    Args:
        case (str): path of case folder (without .tar suffix)

    Returns:
        case (str): path of case folder (without .tar suffix)
    """    
    try:
        # if both the folder and the tar exist, remove the folder and extract tar again (we expect the tar not to be touched)
        if os.path.exists(f'{case}.tar') and os.path.exists(case):
            subprocess.run(f'rm -r {case}', shell=True, check=True)
            return case
        elif os.path.exists(case):
            print('Only the folder of case exists, removing folder')
            subprocess.run(f'rm -r {case}', shell=True, check=True)
        elif os.path.exists(f'{case}.tar'):
            return case
    except subprocess.CalledProcessError as e:
        print('In check_exists: ')
        print(e)

def unpack_archive(case):
    try:
        with tarfile.open(f'{case}.tar', 'r') as tar:
            # need to rename tar members because writing to the original dir will not work otherwise
            for member in tar:
                member.name = os.path.basename(member.name)
            os.mkdir(case)
            tar.extractall(case)
    except tarfile.ReadError as e:
        print(e)
        # remove complete directory so that get_unmodelled cases sees it as unmodelled
        print(f"(unpack archive) Tar file is not valid, removing: {case}")
        subprocess.run(f'rm -r {case}.tar', shell=True, check=True)
        return False
    except subprocess.CalledProcessError as e:
        print('In unpack archive: ')
        print(e)
    except:
        traceback.print_exc()
    assert os.path.exists(case) # if all goes well the tar file should exist
    return True

# cleaning of target directories, to be called in parallel
def clean_target_dir(case:str):
    """clean case directory: remove redundant files/folder produced by PANDORA

    Args:
        case (str): path of case folder without suffix (.tar)
    """    
    for dir_wild in ['/*.DL*', '/*.B99990001.pdb', '/*.V99990001', '/*.D00000001', 
                    '/__pycache__/', '/*.lrsr', '/*.rsr', '/*.sch']:
        wild_paths = glob.glob(f'{case}{dir_wild}')
        for file_path in wild_paths:
            if os.path.exists(file_path):
                try:
                    if dir_wild[-1] == '/':
                        subprocess.run(f'rm -r {file_path}', shell=True, check=True)
                        #subprocess.check_call(f'rm -r {file_path}', shell=True)
                    else:
                        subprocess.run(f'rm {file_path}', shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print('In clean_target_dir: ')
                    print(e)
                except Exception as e:
                    print(e)

def archive_and_remove(case:str):
    """archives the case folder as a .tar file to save inode space

    Args:
        case str: directory name of case to be archived
    """ 
    prefix_case_folder = os.path.split(case.rstrip('/'))[0]
    case_folder = os.path.split(case.rstrip('/'))[1]   
    try:
        subprocess.run(f"tar -cf {case}.tar -C {prefix_case_folder} {case_folder} \
                       --remove-files", shell=True, check=True)
    except subprocess.CalledProcessError as cpe:
        print(f"Something went wrong in archive case: {case}\n{cpe}")
    except Exception as e:
        print(e)
        
def quick_check_cleaned(case:str):
    """quick checking to determine if folder was already cleaned by a previous invocation of this module,
    the check is based on the output of the command line tar -tf output. This check avoids the unneeded opening of the case archive

    Args:
        case (str): path of case folder without suffix (.tar)

    Returns:
        True,False (bool): true if folder is cleaned, false otherwise
    """    
    try:
        output = subprocess.run(f'tar -tf {case}.tar', shell=True, capture_output=True)
        files = output.stdout.decode("ASCII")
        files_list = files.split('\n')
    except subprocess.CalledProcessError as cpe:
        print('Error in quick_check_cleaned\n{cpe}')
        return False
    except Exception as e:
        print(e)
        return False
    to_remove = ['.*\.DL*', '.*.B99990001.pdb', '.*.V99990001', '.*.D00000001', 
                    '.*/__pycache__/.*', '.*.lrsr', '.*.rsr', '.*.sch']
    
    for file in files_list:    
        for file_to_remove in to_remove:
            if re.search(file_to_remove, file):
                return False
    # if no match was found, the case folder is cleaned       
    return True
    

def clean_copy_target_files(sub_folders:list):
    """main process where each case folder goes through a number of steps 
    to be opened, cleaned and archived again

    Args:
        sub_folders (list): list of case folders obtained with glob
    """    
    for folder in sub_folders:
        try:
            case_dir = check_exist(folder)
            # if case was modelled, do the next steps
            if case_dir:
                if quick_check_cleaned(folder):
                    # skip directly to next folder in the forloop
                    continue
                if not unpack_archive(folder): # skip case if files are not valid
                    return 
                clean_target_dir(folder)     
                archive_and_remove(folder)
        except:
                print(f'Cleaning failed for case {folder}')
                print(traceback.format_exc())
                try:
                    # try to archive folder and remove unarchived files after an exception has occured
                    archive_and_remove(folder)
                except:
                    print(traceback.format_exc())

a = arg_parser.parse_args()

# find the paths of the models inside the model_path folder
if a.sequential:
    n_cores = 1
else:
    # path should be string and contain an asterisk
    if type(a.models_dir)!=str or "*" not in a.models_dir:
        raise Exception("Expected a wild card path, please provide a path like this: mymodelsdir/\*/\*")
    # clean the whole models folder
    n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))

wildcard_path = a.models_dir.replace('\\', '')
folders = glob.glob(wildcard_path)
folders = [folder for folder in folders if '.tar' in folder]
folders = [case.split('.')[0] for case in folders]

# run the cleaning
if not a.sequential:
    # create list of lists (inner lists are list of paths), there are n_cores inner lists of approx equal length
    all_paths_lists = []
    chunk = math.ceil(len(folders)/n_cores)
    # cut the process into pieces to prevent spawning too many parallel processing
    for i in range(0, len(folders), chunk):
        all_paths_lists.append(folders[i:min(i+chunk, len(folders))])
    # let each inner list be handled by exactly one thread
    Parallel(n_jobs = n_cores, verbose = 1)(delayed(clean_copy_target_files)(case) for case in all_paths_lists)

else:
    clean_copy_target_files(folders)

print(f"Cleaning finished.")

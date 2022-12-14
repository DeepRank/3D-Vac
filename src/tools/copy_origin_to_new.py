import os
import glob
import argparse
import numpy as np
import subprocess
from joblib import Parallel, delayed

arg_parser = argparse.ArgumentParser(
    description="""
  
    """
)
arg_parser.add_argument("--models-dir", "-d",
    help="Name of the directory where the selected models reside in data/... \
          Should look like: /data/selected_modelels/BA/*",
    type=str
)

def replace_rotated(list_orgs, list_rotated):
    
    for org, rot in zip(list_orgs, list_rotated):
        try:
            subprocess.run(f'cp {org} {rot}', shell=True, check=True)
        except subprocess.CalledProcessError as cpe:
            print(cpe)
            print(traceback.format_exc())
    
def fast_load_dirs_org(sub_folder):
    pdb_files = glob.glob(os.path.join(sub_folder, '*/pdb/*.pdb.origin'))    
    return pdb_files

def fast_load_dirs_rot(sub_folder):
    pdb_files = glob.glob(os.path.join(sub_folder, '*/pdb/*.pdb'))    
    return pdb_files

def run(models_dir):
    
    n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))
    a = arg_parser.parse_args()

    sub_folders = glob.glob(os.path.join(models_dir,'*'))

    print('GLOB')

    pdbs_org_list = Parallel(n_jobs=n_cores, verbose=1)(delayed(fast_load_dirs_org)(pdb_sub) for pdb_sub in sub_folders)
    models_org = [x for sublist in pdbs_org_list for x in sublist]
    
    pdbs_rot_list = Parallel(n_jobs=n_cores, verbose=1)(delayed(fast_load_dirs_rot)(pdb_sub) for pdb_sub in sub_folders)
    models_rot = [x for sublist in pdbs_rot_list for x in sublist]
    
    print(f'COPYING FILES\nnum modififed pdbs: {len(models_rot)}\n num original pdbs: {len(models_org)}')
    pdbs_rot_list = Parallel(n_jobs=n_cores, verbose=1)(delayed(replace_rotated)(org_sub, rot_sub) for org_sub, rot_sub in 
                                                        zip(np.array_split(models_org, n_cores), np.array_split(models_rot, n_cores)))
        


if __name__ == "__main__":
    a = arg_parser.parse_args()
    
    run(a.models_dir)
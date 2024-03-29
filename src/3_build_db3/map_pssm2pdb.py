import argparse
import pandas as pd
import glob
from pathlib import Path
import numpy as np
from pssmgen import PSSM
import os
import math
import traceback
import subprocess
from joblib import Parallel, delayed

arg_parser = argparse.ArgumentParser(
    description = """
    Map the generated raw PSSM to the PDB target structures. 
    """
)
arg_parser.add_argument("--csv-file", "-i",
    help="Name of db1 in data/external/processed/. Default BA_pMHCI.csv",
    default="../../data/external/processed/BA_pMHCI.csv"
)
arg_parser.add_argument("--alphachain-pssm", "-M",
    help="""
    MHC class alpha chain (M chain) raw pssm path containing all mhc alleles
    """,
    required=True
)
arg_parser.add_argument("--betachain-pssm", "-N",
    help="""
    MHC class beta chain (N chain) raw pssm path
    """,
)
arg_parser.add_argument("--mhc-class", "-c",
    help="""
    MHC class
    """,
    default="I",
    choices=["I", "II"],
)

def map_pssms(db2_sub):
    for case in db2_sub:
        work_dir = "/".join(case.split("/")[:-2])
        # check which allele belongs to case
        allele = df[df['ID'] == Path(case).stem.split(".")[0]]['allele'].values[0]
        try:
            if a.mhc_class == 'I':
                chain_M =  glob.glob(os.path.join(a.alphachain_pssm, f'pssm_raw/*{allele}*.pssm'))[0]
                chains = {'M': chain_M}
                
            elif a.mhc_class == 'II':
                allele = allele.split(';')
                # If there is only one allele assign DRA1 as second allele
                if len(allele) == 1:
                    if 'DRB' in allele[0]:
                        alpha_allele = 'HLA-DRA*01:01'
                        beta_allele = allele[0]
                    else:
                        raise Exception(f'ERROR: case {case} has only one allele but does not seem to be HLA-DR')

                # If there are two alleles, assign them to the right chain
                elif len(allele) == 2:
                    alpha_allele = allele[[allele.index(x) for x in allele if 'A' in x.replace('HLA','')][0]]
                    beta_allele = allele[[allele.index(x) for x in allele if 'B' in x.replace('HLA','')][0]]

                chain_M = glob.glob(os.path.join(a.alphachain_pssm, f'pssm_raw/{alpha_allele}.pssm'))[0]
                chain_N = glob.glob(os.path.join(a.betachain_pssm, f'pssm_raw/{beta_allele}.pssm'))[0]
                chains = {'M': chain_M, 'N': chain_N}
        except IndexError as ie:
            print(f'ERROR: Could not find allele of case {case} in pssm db\n{ie}\n{traceback.format_exc()}')
            return
        try:
            subprocess.check_call(f'mkdir {work_dir}/pssm_raw', shell=True)
        except:
            print(f'Warning: could not make pssm_raw folder for case {case}')
        try:
            subprocess.check_call(f'mkdir {work_dir}/pssm', shell=True)
        except:
            print(f'Warning: could not make pssm folder for case {case}')
        model_name = case.split("/")[-1].split('.')[0]
        for chain_id, path in chains.items():
            subprocess.check_call(f'cp {path} {work_dir}/pssm_raw/{model_name}.{chain_id}.pssm', shell=True)

        gen = PSSM(work_dir=work_dir)
        gen.map_pssm(pssm_dir="pssm_raw", pdb_dir="pdb", out_dir="pssm", chain=list(chains.keys()))

def chunk_folders(folders):
        # create list of lists (inner lists are list of paths), there are n_cores inner lists of approx equal length
    all_paths_lists = []
    chunk = math.ceil(len(folders)/n_cores)
    # cut the process into pieces to prevent spawning too many parallel processing
    for i in range(0, len(folders), chunk):
        all_paths_lists.append(folders[i:min(i+chunk, len(folders))])

def fast_load_dirs(sub_folders):
    folder_list = []
    for folder in sub_folders:
        folder_list.extend(glob.glob(os.path.join(folder, '*/pdb/*.pdb')))
    return folder_list


if __name__ == '__main__':
    a = arg_parser.parse_args()
    n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))
    print(f'number of cores {n_cores}')

    csv_path = f"{a.csv_file}"
    df = pd.read_csv(csv_path)

    all_models_sub = glob.glob(f"/projects/0/einf2380/data/pMHC{a.mhc_class}/db2_selected_models/BA/*")
    all_models = Parallel(n_jobs=n_cores, verbose=1)(delayed(fast_load_dirs)(folders_sub) for folders_sub in np.array_split(all_models_sub, n_cores))
    print(f'debug: {type(all_models)}')
    
    total_len = [len(sub) for sub in all_models]
    print(f'Total size of models: {np.sum(total_len)}')

    Parallel(n_jobs=n_cores, verbose=1)(delayed(map_pssms)(db2_sub) for db2_sub in all_models if type(db2_sub) != 'NoneType')

    print(f"Finished mapping {len(all_models)} PSSM to PDB")
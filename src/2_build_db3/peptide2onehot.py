# this script should be runned after map_pssm2pdb.py
import glob
import os
import math
import pandas as pd
import argparse
import numpy as np
from joblib import Parallel, delayed

arg_parser = argparse.ArgumentParser(
    description="""
    This script takes peptides from `peptide` column of db1 (provided with --input-csv parameter) and encode it
    using one-hot encoding to generate a pseudo-PSSM of the peptide.
    The pseudo-PSSM is written into the pssm folder as BA_xyz.P.pssm.
    """
)
arg_parser.add_argument("--input-csv", "-i",
    help="Name of db1 in data/external/processed/. Default BA_pMHCI.csv.",
    default="../../data/external/processed/BA_pMHCI.csv",
)
arg_parser.add_argument("--mhc-class", "-m",
    help="""
    MHC class
    """,
    default="I",
    choices=["I", "II"],
)
arg_parser.add_argument("--models-dir", "-d",
    help="Name of the directory where the selected models reside in data/... \
          Should look like: /data/selected_modelels/BA/*",
    type=str
)

def fast_load_dirs(globpath):
    all_models = []
    all_models_first = glob.glob(globpath)
    for folder in all_models_first:
        all_models.extend(glob.glob(os.path.join(folder, '*')))
    return all_models

def generate_onehot(IDs: np.array):
    """generate onehot encoding for entire peptide sequence set of db2

    Args:
        IDs (numpy.array): indices of all the cases in db2
    """
    failed_cases = []
    succesful_cases = []    
    for _, idx in enumerate(IDs):
        sequence = db2.loc[idx,"peptide"]
        sequence_id = db2.loc[idx, "ID"].replace('_', '-')
        peptide_pssm_rows = [pssm_template]
        for i,res in enumerate(sequence):
            pdbresi = str(i+1)
            pdbresn = res
            seqresi = pdbresi
            seqresn = pdbresn
            peptide_pssm_row = [pdbresi,pdbresn,seqresi,seqresn,*[str(0)]*21]
            onehot_pos = pssm_template.index(res.strip())
            peptide_pssm_row[onehot_pos] = str(1)
            peptide_pssm_rows.append(peptide_pssm_row) 
        #write the file
        search_pssm_path = [path for path in pssm_folders if sequence_id in path.split("/")[-1]]
        if search_pssm_path:
            peptide_pssm_path = search_pssm_path[0] + "/pssm"
        else:
            print(f'ID {sequence_id} is not found in models dir, skipping')
            failed_cases.append(f'{sequence_id} not found in models dir, skipping')
            continue
        if not glob.glob(f"{peptide_pssm_path}/*.M.pdb.pssm"):
            failed_cases.append(f'{sequence_id} M chain pssm not found, cannot create path, skipping')
            continue
        peptide_pssm_file = glob.glob(f"{peptide_pssm_path}/*.M.pdb.pssm")[0].split("/")[-1].replace("M","P")
        peptide_pssm_complete_path = f"{peptide_pssm_path}/{peptide_pssm_file}"
        print(peptide_pssm_complete_path)
        to_write= "\n".join(["\t".join(row) for row in peptide_pssm_rows])
        succesful_cases.append(sequence_id)
        # write the output file
        with open(peptide_pssm_complete_path, "wb") as peptide_f:
            to_write = to_write.encode("utf8").strip()
            peptide_f.write(to_write)
    return succesful_cases, failed_cases
    
if __name__ == "__main__":
    n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))
    a = arg_parser.parse_args()

    pssm_folders = fast_load_dirs(a.models_dir.replace('\\', ''))
    pssm_template_path = "/projects/0/einf2380/data/templates/M_chain_mapped_template.pssm"

    # retrieve the first row of the pssm_template to make the template for the pseudo-PSSM
    pssm_template = []
    with open(pssm_template_path) as template_f:
        rows = [row.replace("\n", "").split() for row in template_f]
        pssm_template = rows[0]

    # make the peptide_sequences
    db2 = pd.read_csv(f"{a.input_csv}")
    # IDs = comm.scatter(IDs, root=0)
    IDs = np.array(list(range(len(db2))))

    all_ids_lists = []
    print(f'len db2 {len(IDs)}')
    print(f'n_cores {n_cores}')
    chunk = math.ceil(len(IDs)/n_cores)
    # cut the process into pieces to prevent spawning too many parallel processing
    for i in range(0, len(IDs), chunk):
        all_ids_lists.append(IDs[i:min(i+chunk, len(IDs))])
    # let each inner list be handled by exactly one thread
    succesful_cases, failed_cases = zip(*Parallel(n_jobs = n_cores, verbose = 1)(delayed(generate_onehot)(case_id) for case_id in all_ids_lists))
    
    failed_cases_list = np.array(failed_cases, dtype=object).flatten().tolist()
    failed_cases_format = '\n'.join('\n'.join(l) for l in failed_cases_list)
    
    succesful_cases_list = np.array(succesful_cases, dtype=object).flatten().tolist()
    succesful_cases_format = '\n'.join('\n'.join(l) for l in succesful_cases_list)
    
    print(f'succesful_cases: {succesful_cases_format}\n\n failed_cases: {failed_cases_format}')

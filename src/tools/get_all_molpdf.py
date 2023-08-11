import tarfile
from joblib import Parallel, delayed 
import os
from modeller import *
from modeller.scripts import complete_pdb
import pandas as pd
import glob
import math

def extract_member(case, member_name):
    try:
        with tarfile.open(f'{case}.tar', 'r') as tar:
            # need to rename tar members because writing to the original dir will not work otherwise
            for member in tar:
                member.name = os.path.basename(member.name)

            case_path = os.path.join(temp, os.path.basename(case))
            try:
                if not os.path.exists(case_path):
                    os.mkdir(case_path)
            except FileExistsError:
                print("folder already made")
            tar.extract(member_name, os.path.join(temp, os.path.basename(case)))
        # return the full path of tar member
        return os.path.join(temp, os.path.basename(case), member_name)
    
    except tarfile.ReadError as e:
        print(e)
        # remove complete directory so that get_unmodelled cases sees it as unmodelled
        print(f"(extract_member) Tar file is not valid, removing: {case}")
        return False
    except KeyError as ke:
        print(f'In extract_member: molpdf not found for {case}:\n{ke}')
    except subprocess.CalledProcessError as e:
        print('In extract_member: ')
        print(e)
    except:
        traceback.print_exc()# if all goes well the tar file should exist

# def retrieve_best_model(case):
#     molpdf_path = extract_member(case, "molpdf_DOPE.tsv")
#     if molpdf_path:
#         try:
#             molpdf_df = pd.read_csv(molpdf_path, sep="\t", header=None)
#         except pd.errors.EmptyDataError:
#             print(f"empty df: {molpdf_path}")
#             return
#     else:
#         # go back to the run function to break processing case and print traceback
#         raise Exception
#     target_scores = molpdf_df.iloc[:,1].sort_values()[:1]
#     target_mask = [score in target_scores.tolist() for score in molpdf_df.iloc[:,1]]
#     target_ids = molpdf_df[target_mask].iloc[:,0]
#     target = [f"{case}/{structure}" for structure in target_ids][0]
#     return target

def assess_model(case):
    molpdf_path = extract_member(case, "molpdf_DOPE.tsv")
    with open(f'{case}/molpdf_DOPE.tsv') as infile:
        best_model = infile.readline().split('\t')[0]
        
    best_model_path = extract_member(case, best_model)
    env = environ()
    env.libs.topology.read(file='$(LIB)/top_heav.lib')
    env.libs.parameters.read(file='$(LIB)/par.lib')
    
    mdl = complete_pdb(env, f'{case}/{best_model_path}')
    score = mdl.assess_normalized_dope()
    return score

def run(case_paths):
    for case in case_paths:
        try:
            #target = retrieve_best_model(case)
            score = assess_model(case)
            return (case, targets)
        except:
            print(traceback.format_exc())
            print(f'Copying best models of case: {case} not succesful')
            return (case, None)

if __name__ == '__main__':
    #a = arg_parser.parse_args()

    #n_cores = int(os.getenv('SLURM_CPUS_ON_NODE'))
    n_cores = 4
    models_path = '/projects/0/einf2380/data/pMHCI/3d_models/BA/\*/\*'
    csv_path = '/projects/0/einf2380/data/external/processed/I/BA_pMHCI_human_quantitative.csv'
    #db2_selected_models_path = a.best_models_path
    # TMP dir to write temporary files to 
    base_tmp = os.environ["TMPDIR"]
    temp = os.path.join(base_tmp, "db3_copy_3Dmodels")
    if not os.path.exists(temp):
        os.mkdir(temp)

    # do a check if models dir is passed in the correct way
    if "*" not in models_path and type(models_path)!=list:
        print("Expected a wild card path, please provide a path like this: mymodelsdir/\*/\*")
        raise SystemExit

    # read the csv from db2 to find all case ids
    df = pd.read_csv(csv_path, header=0)
    df['ID'] = df['ID'].apply(lambda x: '_'.join(x.split('-')))

    wildcard_path = models_path.replace('\\', '')
    folders = glob.glob(wildcard_path)
    folders = [folder for folder in folders if '.tar' in folder]
    all_models = [case.split('.')[0] for case in folders]
    # filter out the models that match in the original csv from db2
    db2 = [folder for folder in all_models if folder.split("/")[-1] in df["ID"].tolist()]
    db2 = all_models
    

    all_paths_lists = []
    print(f'len db2 {len(db2)}')
    print(f'n_cores {n_cores}')
    chunk = math.ceil(len(db2)/n_cores)
    # cut the process into pieces to prevent spawning too many parallel processing
    for i in range(0, len(db2), chunk):
        all_paths_lists.append(db2[i:min(i+chunk, len(db2))])
    # let each inner list be handled by exactly one thread
    raise Exception('ok')
    cases = Parallel(n_jobs = n_cores, verbose = 1)(delayed(run)(case) for case in all_paths_lists)
    
    with open(a.outpkl, 'wb') as outpkl:
        pickle.dump(cases, outpkl)
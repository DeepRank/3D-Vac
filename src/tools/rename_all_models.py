import subprocess 
from glob import glob
from joblib import Parallel, delayed


def rename(folder): 
    for f in glob(folder + '/*/*'): 
        nf = f.split('.') 
        nf.pop(1) 
        nf = ('.').join(nf).replace('-','_') 
        subprocess.check_call(f'mv {f} {nf}', shell=True) 


folders = glob('/projects/0/einf2380/data/pMHCII/db2_selected_models/BA/*/*') 
Parallel(n_jobs = 64, verbose = 1)(delayed(rename)(infolder) for infolder in folders)
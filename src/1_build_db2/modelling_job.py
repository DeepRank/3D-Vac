import sys
import time
import os
from PANDORA.Wrapper import Wrapper
from PANDORA.Database import Database
from math import ceil
import multiprocessing
import numpy as np
import pandas as pd
import argparse


arg_parser = argparse.ArgumentParser(
    description="Performs 3D-modelling of the cases provided on one node."
)
arg_parser.add_argument("--mhc-class", "-m",
    help="MHC class of the cases",
    choices=['I','II'],
    required=True,
)
arg_parser.add_argument("--csv-path", "-c",
    help="Path to csv file with the cases to be modelled.",
    required=True
)
arg_parser.add_argument("--running-time", "-t",
    help="Running time",
    required=True
)
arg_parser.add_argument("--num-cores", "-n",
    help="Number of cores to be used per node.",
    default=128
)
arg_parser.add_argument("--db-path", "-d",
    help="Path to PANDORA database folder.",
    default="/projects/0/einf2380/softwares/PANDORA/PANDORA_files/data/csv_pkl_files/20220708_complete_db.pkl"
)
a = arg_parser.parse_args()

print(f'INFO: \n cases per hour per node :{10*a.num_cores} \n num of cores: {a.num_cores}\n \
running time:{int(a.running_time)}\nbatch: {10*a.num_cores*int(a.running_time)}')

# determine node index so we don't do the same chunk multiple times
node_index = int(os.getenv('SLURM_NODEID'))

# check if there are cases to model
df = pd.read_csv(f"{a.csv_path}")
if df.empty:
    print('No new cases to model, exiting')
    sys.exit(0)


cases_per_hour_per_node = 10*int(a.num_cores) # 10*128= 1280
batch = cases_per_hour_per_node*int(a.running_time)

start_row = (batch*node_index)
end_row = (batch*node_index) + batch
# it is possible that the end row index exceeds the length of the file because the file length is not divisible by the batch size
if end_row > df.shape[0]+1: 
    end_row = df.shape[0]+1 

print(f'INFO: Node index: {node_index} \nStart row: {start_row} \nEnd row: {end_row} \nBatch size: {batch}')

# Load the database file
print('Loading Database..')
db = Database.load(a.db_path)
print('Database loaded')
#db.update_ref_sequences()
PDB_path = a.db_path.split('/data/')[0] + '/data/PDBs'
db.repath(PDB_path, save=False)
print('Database repathed')            


#find outdir column
outdir_col = df.columns.to_list().index('db2_folder')

# DEBUG
print(f"INFO:\n path: {a.csv_path}\n db: {a.db_path}\n MHC_class={a.mhc_class}\n outdir_col={outdir_col} start_row={start_row}, end_row={end_row}\n num_cores={a.num_cores}")

#Create targets
t1 = time.time()
wrap = Wrapper.Wrapper()
wrap.create_targets(a.csv_path, db, 
    MHC_class=a.mhc_class, header=True, delimiter=',', IDs_col=0,
    peptides_col=2, allele_col=1, outdir_col=outdir_col, benchmark=False, 
    verbose=True, start_row=start_row, end_row=end_row, use_netmhcpan=True
)
t2 = time.time()
print('Wrapper created')
print(f"Time to predict anchors: {t2-t1}")

# Run the models
wrap.run_pandora(num_cores=a.num_cores, n_loop_models=20, clip_C_domain=True, 
    benchmark=False)
t3 = time.time()
print(f"Time to model: {t3-t2}")

wrapping_time = t2-t1
modelling_time = t3-t2

# wrapping_times = comm.gather(wrapping_time)
# modelling_times = comm.gather(modelling_time)

# if rank==0:
#     print("Average time to create wrappers: ", float(np.array(wrapping_times).mean()))
#     print("Average time to create models: ", float(np.array(modelling_times).mean()))
import sys
import time
from PANDORA.Wrapper import Wrapper
from PANDORA.Database import Database
from math import ceil
from mpi4py import MPI
import multiprocessing
import numpy as np
import pandas as pd
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
running time:{int(a.running_time)}\n rank:{rank} \n batch: {10*a.num_cores*int(a.running_time)}')

# total number of cases per hour for each node: (3600/(time for modeling a case for a core))*num_cores
cases_per_hour_per_node = 10*int(a.num_cores) # 1536
batch = cases_per_hour_per_node*int(a.running_time)
start_row = int(rank*batch)

end_row = int((rank+1)*batch)
print(f"Rank {rank}. start_row: {start_row} end_row: {end_row}. Number of cores: {multiprocessing.cpu_count()}")

# Load the database file
print('Loading Database..')
db = Database.load(a.db_path)
print('Database loaded')
#db.update_ref_sequences()
PDB_path = a.db_path.split('/data/')[0] + '/data/PDBs'
db.repath(PDB_path, save=False)
print('Database repathed')            

# check if there are cases to model
df = pd.read_csv(f"{a.csv_path}")
if df.empty:
    print('No new cases to model, exiting')
    sys.exit(0)

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

wrapping_times = comm.gather(wrapping_time)
modelling_times = comm.gather(modelling_time)

if rank==0:
    print("Average time to create wrappers: ", float(np.array(wrapping_times).mean()))
    print("Average time to create models: ", float(np.array(modelling_times).mean()))
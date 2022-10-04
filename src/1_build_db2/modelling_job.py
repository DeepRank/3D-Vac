import sys
import time
import os
from PANDORA.Wrapper import Wrapper
from PANDORA.Database import Database
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
    type=int,
    default=128
)
arg_parser.add_argument("--batch-size", "-b",
    help="Batch size calculated by allocate_nodes.py.",
    required=True
)

a = arg_parser.parse_args()
CASES_PER_HOUR_PER_CORE = 10

print(f'INFO: \n cases per hour per node :{CASES_PER_HOUR_PER_CORE*a.num_cores} \n num of cores: {a.num_cores}\n \
running time:{a.running_time}\nbatch: {a.batch_size}')

# determine node index so we don't do the same chunk multiple times
node_index = int(os.getenv('SLURM_NODEID'))

# check if there are cases to model
df = pd.read_csv(f"{a.csv_path}")
if df.empty:
    print('No new cases to model, exiting')
    sys.exit(0)


start_row = (int(a.batch_size)*node_index)
end_row = (int(a.batch_size)*node_index) + int(a.batch_size)
# it is possible that the end row index exceeds the length of the file because the file length is not divisible by the batch size
if end_row > df.shape[0]: 
    end_row = df.shape[0] 

print(f'INFO: Node index: {node_index} \nStart row: {start_row} \nEnd row: {end_row} \nBatch size: {a.batch_size}')

# Load the database file
print('Loading Database..')
db = Database.load()
print('Database loaded')

#find outdir column
outdir_col = df.columns.to_list().index('db2_folder')

# DEBUG
print(f"INFO:\n path: {a.csv_path}\n MHC_class={a.mhc_class}\n outdir_col={outdir_col} start_row={start_row}, end_row={end_row}\n num_cores={a.num_cores}")

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
    benchmark=False, archive=True)
t3 = time.time()
print(f"Time to model: {t3-t2}")

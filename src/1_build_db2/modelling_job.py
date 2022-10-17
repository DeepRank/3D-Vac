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
arg_parser.add_argument("--n-structures", "-s",
    help="Number of structures to let PANDORA model",
    type=int,
    default=20,
)
arg_parser.add_argument("--node-index", "-i",
    help="Node id used for parallelization, when not passed serial case is assumed.",
    required=False,
    default= 0
)

a = arg_parser.parse_args()
CASES_PER_HOUR_PER_CORE = 10

print(f'INFO: \n cases per hour per node :{CASES_PER_HOUR_PER_CORE*a.num_cores} \n num of cores: {a.num_cores}\n \
running time:{a.running_time}\nbatch: {a.batch_size}')

# determine node index so we don't do the same chunk multiple times
# node_index = int(os.getenv('SLURM_NODEID'))
node_index = int(a.node_index)
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

t2 = time.time()

## B. Create all Target Objects based on peptides in the .tsv file
wrap = Wrapper.Wrapper(a.csv_path, db, MHC_class=a.mhc_class, 
                    IDs_col=0, peptides_col=2, allele_name_col=1,
                    outdir_col=outdir_col, archive=True,
                    benchmark=True, verbose=True, delimiter=',',
                    header=True, num_cores=a.num_cores, use_netmhcpan=True,
                    n_loop_models=a.n_structures, clip_C_domain=True,
                    start_row=start_row, end_row=end_row)

t3 = time.time()
print(f"Time to model: {t3-t2}")

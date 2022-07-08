import sys
sys.path.append('/home/lepikhovd/softwares/PANDORA/')
import time
from PANDORA.Wrapper import Wrapper
from PANDORA.Database import Database
from math import ceil
from mpi4py import MPI
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

csv_path = "../../data/external/processed/to_model.csv"
df = pd.read_csv(csv_path)

running_time = int(sys.argv[1])

num_cores = 128
#total number of cases per hour for each node: (3600/(time for modeling a case for a core))*num_cores
cases_per_hour_per_node = 10*num_cores # 1536
batch = cases_per_hour_per_node*running_time
start_row = int(rank*batch)

if rank == size-1:
    end_row = len(df)
else:
    end_row = int((rank+1)*batch)
# print(f"Rank {rank}. start_row: {start_row} end_row: {end_row}. Number of cores: {multiprocessing.cpu_count()}")

#Load the database file
print('Loading Database..')
db = Database.load("/home/lepikhovd/softwares/PANDORA/data/csv_pkl_files/complete_db_20221904.pkl")
print('Database loaded')
#db.update_ref_sequences()

db.repath('/home/lepikhovd/softwares/PANDORA/data/PDBs', save=False)
print('Database repathed')            

#Create targets
t1 = time.time()
wrap = Wrapper.Wrapper()
wrap.create_targets(csv_path, db, 
    MHC_class='I', header=True, delimiter=',', IDs_col=8, 
    peptides_col=1, allele_col=0, outdir_col=-1, benchmark=False, verbose=False,
    start_row=start_row, end_row=end_row, use_netmhcpan=True
)
t2 = time.time()
print('Wrapper created')
## Run the models
wrap.run_pandora(num_cores=num_cores, n_loop_models=20, 
    benchmark=False)
t3 = time.time()
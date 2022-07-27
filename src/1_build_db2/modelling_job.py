import sys
sys.path.append('/home/lepikhovd/softwares/PANDORA/')
import time
from PANDORA.Wrapper import Wrapper
from PANDORA.Database import Database
from math import ceil
from mpi4py import MPI
import multiprocessing
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

csv_path = "../../data/external/processed/to_model.csv"

running_time = int(sys.argv[1])

num_cores = 128
# total number of cases per hour for each node: (3600/(time for modeling a case for a core))*num_cores
cases_per_hour_per_node = 10*num_cores # 1536
batch = cases_per_hour_per_node*running_time
start_row = int(rank*batch)

end_row = int((rank+1)*batch)
print(f"Rank {rank}. start_row: {start_row} end_row: {end_row}. Number of cores: {multiprocessing.cpu_count()}")

# Load the database file
print('Loading Database..')
db = Database.load("/home/lepikhovd/softwares/PANDORA/data/complete_db_08_06_2022.pkl")
print('Database loaded')
#db.update_ref_sequences()

db.repath('/home/lepikhovd/softwares/PANDORA/PANDORA_files/data/PDBs', save=False)
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
print(f"Time to predict anchors: {t2-t1}")

# Run the models
wrap.run_pandora(num_cores=num_cores, n_loop_models=20, clip_C_domain=True, 
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
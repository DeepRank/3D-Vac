import sys
sys.path.append('/home/lepikhovd/softwares/PANDORA/')
import time
from PANDORA.Wrapper import Wrapper
from PANDORA.Database import Database
from math import ceil
from mpi4py import MPI
import multiprocessing

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

csv_path = "/home/lepikhovd/3D-Vac/data/binding_data/to_model.csv"
csv_file = open(csv_path, "r")

running_time = int(sys.argv[1])

num_cores = 128
#total number of cases per hour for each node: (3600/(time for modeling a case for a core))*num_cores
cases_per_hour_per_node = 12*num_cores # 1536
batch = cases_per_hour_per_node*running_time
start_row = int(rank*batch)
end_row = int((rank+1)*batch)

# if rank != size-1:
#     end_row = int((rank+1)*batch)
# else:
#     n_case_lines = len([line for line in csv_file])
#     end_row = n_case_lines 
# output_dir = '/projects/0/einf2380/data/pMHCI/models/temp'
print(f"Rank {rank}. start_row: {start_row} end_row: {end_row}. Number of cores: {multiprocessing.cpu_count()}")

# #Load the database file
# print('Loading Database..')
# db = Database.load("/home/lepikhovd/softwares/PANDORA/data/csv_pkl_files/complete_db_20221904.pkl")
# print('Database loaded')
# #db.update_ref_sequences()

# db.repath('/home/lepikhovd/softwares/PANDORA/data/PDBs', save=False)
# print('Database repathed')            

# #Create targets
# t1 = time.time()
# wrap = Wrapper.Wrapper()
# wrap.create_targets(csv_path, db, 
#                     MHC_class='I', header=False, delimiter=',', IDs_col=0, 
#                     peptides_col=2, allele_col=1, benchmark=False, verbose=False,
#                     start_row=start_row, end_row=end_row, use_netmhcpan=True)
# t2 = time.time()
# print('Wrapper created')
# ## Run the models
# wrap.run_pandora(num_cores=128, n_loop_models=20, 
#                     benchmark=False)
# t3 = time.time()
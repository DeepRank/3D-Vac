import sys
import time
from PANDORA.Wrapper import Wrapper
from PANDORA.Database import Database
from math import ceil
import json
import random
import string

n_cores = int(sys.argv[1]);
start_row = int(sys.argv[2]);
end_row = int(sys.argv[3]);

output_dir = '/projects/0/einf2380/data/pMHCI/models/temp'

t0 = time.time()

#Load the database file
print('Loading Database..')
db = Database.load("/home/lepikhovd/softwares/PANDORA/data/csv_pkl_files/database.pkl")
print('Database loaded')
#db.update_ref_sequences()

db.repath('/home/lepikhovd/softwares/PANDORA/data/PDBs', save=False)
print('Database repathed')            

#Create targets
t1 = time.time()
wrap = Wrapper.Wrapper()
csv_file = "/home/lepikhovd/binding_data/to_model.csv"
wrap.create_targets(csv_file, db, 
                    MHC_class='I', header=False, delimiter=',', IDs_col=0, 
                    peptides_col=2, allele_col=1, benchmark=False, verbose=False,
                    start_row=start_row, end_row=end_row, use_netmhcpan=True)
t2 = time.time()
print('Wrapper created')
## Run the models
wrap.run_pandora(num_cores=n_cores, n_loop_models=20, 
                    benchmark=False, output_dir=output_dir)
t3 = time.time()

modelling_time = t3-t2;
per_node_time = (modelling_time/(end_row-start_row))*n_cores;
db_time = t1-t0
wrapper_time = t2-t1
number_of_cases = end_row - start_row

letters = string.ascii_lowercase
ran_str = "".join(random.choice(letters) for i in range(5));
with open(f"/projects/0/einf2380/data/modelling_logs/wrapper_output_{ran_str}.json", "w") as f:
    entries = {};
    entries["db_time"] = db_time;
    entries["per_node_time"] = per_node_time;
    entries["creating_wrapper_targets"] = wrapper_time;
    entries["modeling_time"] = modelling_time;
    entries["number_of_cases"] = number_of_cases;
    to_write = json.dumps(entries);
    f.write(to_write)

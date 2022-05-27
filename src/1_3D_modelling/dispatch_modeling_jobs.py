import subprocess
import re
from math import ceil
import argparse

arg_parser = argparse.ArgumentParser(description="Script to dispatch jobs on the snellius cluster. The running time is set to 1 hour. To time has to be hardcoded into the script to be changed, as well as in the `submit_jobs_snellius.sh` file")
arg_parser.add_argument("--cases", "-c",
    help="number of cases to model, in total. Can be an aproximative value. Always greater than the actual number of cases. This can be easily determined with `wc -l <path to your .csv file>`",
    required=True,
    type=int
)
a = arg_parser.parse_args();

running_time = 1; #in hour
num_cores = 128;
#total number of cases per hour for each node: (3600/(time for modeling a case for a core))*num_cores
cases_per_hour_per_node = 12*num_cores # 1536
batch = ceil(running_time*cases_per_hour_per_node)
n_jobs = ceil(a.cases/batch)

command_output = subprocess.check_output(["sbatch", "get_unmodelled_cases.sh"]).decode("ASCII");
jid = int(re.search(r"\d+", command_output).group())

start_line = 0
end_line = start_line+batch
for j in range(n_jobs):
    command_output = subprocess.call(["sbatch",
    f"--dependency=afterany:{jid}",
    "submit_jobs_snellius.sh", 
    str(start_line), 
    str(end_line),
    ]);
    start_line = end_line;
    end_line = start_line + batch
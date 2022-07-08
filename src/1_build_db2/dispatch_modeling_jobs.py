import subprocess
import re
from math import ceil
import argparse

arg_parser = argparse.ArgumentParser(
    description="Script to dispatch jobs on the snellius cluster. \
        The job runs for a defined period of time on an allocated number of clusters. \
        When choosing the running time and the number of nodes, it should be known that each node can \
        model around 1500 cases per hour.")
arg_parser.add_argument("--running-time", "-t",
    help="Number of hours allocated for the job to run. Should be in format 00. Default 01.",
    default="01",
)
arg_parser.add_argument("--jobs", "-n",
    help="Number of jobs run parallely (MPI tasks). Each node has 1 task using all CPU power. The number of nodes \
    is equal to the number of tasks.",
    default=7,
    type=int
)
arg_parser.add_argument("--input-csv", "-i",
    help="db1 file name in data/external/processed. Required.",
    required=True,
)
arg_parser.add_argument("--skip-check", "-s",
    help="Skip the verification of unmodelled cases. By default, if this argument is not provided, already checked models \
    are checked.",
    default=False,
    action="store_true"
)
a = arg_parser.parse_args();


running_time = a.running_time; #in hour

# generate the to_model.csv containing all unmodelled cases from the input csv.  

if a.skip_check == False:
    command_output = subprocess.check_output(
        [
            "sbatch", 
            "get_unmodelled_cases.sh",
            "-f", a.input_csv
        ]
    ).decode("ASCII");

    jid = int(re.search(r"\d+", command_output).group())

    # run the parallel modeling on n nodes
    subprocess.run(
        [
            "sbatch",
            f"--nodes={a.jobs}",
            f"--time={running_time}:00:00",
            f"--dependency=afterany:{jid}",
            "submit_job_snellius.sh",
            running_time,
        ]
    )
else: 
    subprocess.run(
        [
            "sbatch",
            f"--nodes={a.jobs}",
            f"--time={running_time}:00:00",
            "submit_job_snellius.sh",
            running_time,
        ]
    )
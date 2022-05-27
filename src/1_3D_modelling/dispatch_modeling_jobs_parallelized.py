import subprocess
import re
from math import ceil
import argparse

arg_parser = argparse.ArgumentParser(
    description="Script to dispatch jobs on the snellius cluster. \
        The job runs for a defined period of time on an allocated number of clusters. \
        When choosing the running time and the number of nodes, it should be known that each node can \
        model around 1500 cases per hour")
arg_parser.add_argument("--running-time", "-t",
    help="Number of hours allocated for the job to run. Should be in format 00.",
    default="01",
)
arg_parser.add_argument("--nodes", "-n",
    help="Number of nodes for the job. For MPI, this is the number of tasks as well. Each node has 1 task, so number of nodes \
    is equal to the number of tasks.",
    default=7,
    type=int
)
a = arg_parser.parse_args();


running_time = a.running_time; #in hour

# command_output = subprocess.check_output(["sbatch", "get_unmodelled_cases.sh", a.input_csv]).decode("ASCII");
# jid = int(re.search(r"\d+", command_output).group())

subprocess.run([
    "sbatch",
    f"--nodes={a.nodes}",
    f"--time={running_time}:00:00",
    # f"--dependency=afterany:{jid}",
    "submit_job_snellius.sh",
    str(a.nodes),
    running_time,
])
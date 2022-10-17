import subprocess
import re
import argparse

arg_parser = argparse.ArgumentParser(
    description = "Script to dispatch parallelized job on the snellius cluster modelling pMHC complexes from \
    data/externa/processed/to_model.csv (generated by `get_unmodelled_cases.py`). \
    This script first calls `unmodelled_cases.py` to generate the `data/external/processed/to_model.csv` file \
    containing unmodelled cases present in the --input-csv. Then, `allocate_nodes.py` is called to calculate the \
    necessary number of nodes (runned in parallel) based on the running time. \
    The `modelling_job.py` runs for a defined period of time on an allocated number of clusters. \
    At the end of the modelling job, `clean_outputs.py` is called to clean the models from unecessary files.\
    When choosing the running time, it should be known that each core on each node can \
    model around 1280 cases per hour. The number of nodes to allocate for this job is calculated by allocate_nodes.py \
    w.r.t the number of cases per hour per core. Because the anchors are predicted using NetMHCPpan4.1 tool. \
    \nUsage\n: You can either specify number of nodes [num-nodes=n]. This will calculate the running time per nodes based \
     on the number of cases. OR you can specify a running time [running-time=00] in hours. This will calculate the number \
    of nodes needes to process the number of cases based on the running time. These calculations are done in allocate_nodes.py")
arg_parser.add_argument("--running-time", "-t",
    help = "Number of hours allocated for the job to run. Should be in format 00. Default 01.",
    default = "01",
)
arg_parser.add_argument("--num-nodes", "-n",
    help = "Number of nodes to use. Should be in format 00. Default 0 (will use running-time instead).",
    default = "00",
)
arg_parser.add_argument("--input-csv", "-i",
    help = "db1 file path. No default value. Required.",
    required = True,
)
arg_parser.add_argument("--skip-check", "-c",
    help = "Skip the verification of unmodelled cases. By default, if this argument is not provided, models \
    are checked.",
    default = False,
    action = "store_true"
)
arg_parser.add_argument("--models-dir", "-d",
    help="Path to the BA or EL folder where the models are generated",
    default="/projects/0/einf2380/data/pMHCI/3D_models/BA",
)
arg_parser.add_argument("--mhc-class", "-m",
    help="MHC class of the cases",
    choices=['I','II'],
    required=True,
)
arg_parser.add_argument("--n-structures", "-s",
    help="Number of structures to let PANDORA model",
    default=20,
)
a = arg_parser.parse_args();


running_time = a.running_time; #in hour
to_model = ('/').join(a.input_csv.split('/')[:-1]) + "/to_model.csv"

# generate the to_model.csv containing all unmodelled cases from the input csv.  

if a.skip_check == False:
    command_output = subprocess.check_output(
        [
            "sbatch", 
            "get_unmodelled_cases.sh",
            "--csv-file", a.input_csv,
            "--update-csv", # this argument is mandatory to overwrite `to_model.csv`
            "--models-dir", a.models_dir,
            "--to-model", to_model,
            '--parallel',
            "--archived"
        ]
    ).decode("ASCII");

    jid_get_unmod = int(re.search(r"\d+", command_output).group())

    # run the parallel modeling on n nodes
    subprocess.run(
        [
            "sbatch",
            f"--dependency=afterok:{jid_get_unmod}",
            "allocate_nodes.sh",
            "--running-time", running_time,
            "--num-nodes", a.num_nodes,
            "--mhc-class", a.mhc_class,
            "--input-csv", to_model,
            "--models-dir", a.models_dir,
            "--n_structures", a.n_structures
        ]
    )
else: 
    subprocess.run(
        [
            "sbatch",
            "allocate_nodes.sh",
            "--running-time", running_time,
            "--num-nodes", a.num_nodes,
            "--mhc-class", a.mhc_class,
            "--input-csv", to_model,
            "--models-dir", a.models_dir,
            "--n_structures", a.n_structures
        ]
    )
import argparse
import glob
import pandas as pd
import tarfile
import re
import multiprocessing as mp
import numpy as np

arg_parser = argparse.ArgumentParser(description="""
    Adds an anchor column to the db1.
""")

arg_parser.add_argument("--csv-file", "-f",
    help="Path to db1.",
    default="../../data/external/processed/all_hla0201.csv"
)
arg_parser.add_argument("--models-path", "-p",
    help="Path to newly generated db2.",
    default="/projects/0/einf2380/data/pMHCI/3d_models/BA/*/*"
)
arg_parser.add_argument("--n-jobs", "-n",
    help="Number of worker for the multiprocessing.Pool. Default 4",
    default=4,
    type=int
)

a = arg_parser.parse_args()
df = pd.read_csv(a.csv_file)

# get all models:
models = glob.glob(a.models_path)
models_id = [model.split("/")[-1].replace(".tar","") for model in models]
id_model = dict(zip(models_id, models))

# define the function as a task for the mp.Pool
def assign_anchors(ids):
    id_anchors = {}
    for case_id in ids:
        if case_id in models_id:
            archive = tarfile.open(id_model[case_id])
        else:
            print(f"{case_id} is not modelled.")
            continue
        f = archive.extractfile(f"{case_id}/MyLoop.py")
        lines = [l for l in f]
        anchors = re.findall("\d+", str(lines[17]))
        anchors[0] = int(anchors[0]) - 1
        anchors[-1] = int(anchors[-1]) + 1
        id_anchors[case_id] = anchors
        f.close()
    return id_anchors

if __name__ == "__main__":
    # initialize process pool:
    pool = mp.Pool(a.n_jobs)
    case_ids = [case for case in df["ID"]]
    case_ids = np.array(case_ids)
    case_ids = np.array_split(case_ids, a.n_jobs)
    res = pool.map(assign_anchors, case_ids)
    res = {list(item.keys())[0]:list(item.values())[0] for item in res}

    for case_id,anchors in res.items():
        df.loc[df["ID"] == case_id, "anchors"] = str(anchors)
    df.to_csv(header=False)
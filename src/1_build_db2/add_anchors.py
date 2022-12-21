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
            id_anchors[case_id]= None
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
    case_ids = df["ID"].tolist()
    case_ids = np.array(case_ids)
    case_ids = np.array_split(case_ids, a.n_jobs)
    res = pool.map(assign_anchors, case_ids)
     # make a flat dictionary out of the list of dictionaries
    res = {list(item.keys())[i]:list(item.values())[i] for item in res for i in range(len(list(item.keys())))}
    # transform to a pandas series
    anchor_series = pd.DataFrame.from_dict(res, orient='index', columns=['anchor_0' , 'anchor_1'])
    # rest the index to make concat possible
    df.set_index(keys='ID', inplace=True)
    # remove old anchors:
    if "anchor_0" in df.columns.tolist():
        df.drop(["anchor_0", "anchor_1"], axis=1, inplace=True)
    # concatenate the new series(with anchors) to the orignal dataframe
    df_with_anchors = pd.concat([df, anchor_series], axis=1)
    # write the file (with index, because these are the IDs)
    df_with_anchors["ID"] = df.index
    df_with_anchors.to_csv(a.csv_file, index=False)
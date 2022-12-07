import argparse
import glob
import pandas as pd
import tarfile
import re

arg_parser = argparse.ArgumentParser(description="""
    Adds an anchor column to the db1.
""")

arg_parser.add_argument("--csv-file", "-f",
    help="Path to db1.",
    default="../../data/external/processed/BA_pMHCI.csv"
)
arg_parser.add_argument("--models-path", "-p",
    help="Path to newly generated db2.",
    default="/projects/0/einf2380/data/pMHCI/3d_models/BA/*/*"
)

a = arg_parser.parse_args()
df = pd.read_csv(a.csv_file)

# get all models:
models = glob.glob(a.models_path)
models_id = [model.split("/")[-1].replace(".tar","") for model in models]
id_model = dict(zip(models_id, models))

for i,case_id in enumerate(df["ID"]):
    if i == 0:
        case_id = case_id.replace("_","-")
        print(id_model[case_id])
        archive = tarfile.open(id_model[case_id])
        f = archive.extractfile(f"{case_id}/MyLoop.py")
        lines = [l for l in f]
        anchors = re.findall("\d", str(lines[17]))
        df.loc[df["ID"] == case_id.replace("-","_"), "anchors"] = str(anchors)
        print(df.loc[df["ID"] == case_id.replace("-", "_"), "anchors"])
    else:
        break
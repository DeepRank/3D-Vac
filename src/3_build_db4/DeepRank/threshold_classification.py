import pandas as pd
import numpy as np
import os


def __compute_target__(pdb, targrp):
    """
    Classifies the model from targrp as a binder or non binder based
    on the threshold value from csv.
    Args:
    csv_path: path of the db1 csv
    threshold: plateau to define binders (should be 500)
    """
    #in_csv = os.environ['TARGET_INPUT_CSV']
    in_csv="/projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv"
    df = pd.read_csv(in_csv)
    tarname = "BIN_CLASS"
    molname = targrp.parent.name.replace("/", "")
    class_id = (0.,1.)[ df[df["ID"] == molname]["measurement_value"].values[0] < 500. ]
    print(np.array(class_id).shape)
    targrp.create_dataset("BIN_CLASS", data=np.array(class_id))
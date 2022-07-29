import pandas as pd
import numpy as np


def __compute_target__(pdb, targrp):
    """
    Classifies the model from targrp as a binder or non binder based
    on the threshold value from csv.
    Args:
    csv_path: path of the db1 csv
    threshold: plateau to define binders (should be 500)
    """
    df = pd.read_csv("/home/lepikhovd/training_branch_3D_vac/data/external/processed/BA_pMHCI.csv")
    tarname = "BIN_CLASS"
    molname = targrp.parent.name.replace("/", "")
    class_id = (0.,1.)[ df[df["ID"] == molname]["measurement_value"].values[0] < 500. ] 
    print(np.array(class_id).shape)
    # targrp.create_dataset("BIN_CLASS", data=np.array(class_id))
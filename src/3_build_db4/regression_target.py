import pandas as pd
import numpy as np
import os

def normalize(values):
    """Normalizes the BA values between 0 and 1. Change this function to change the
        normalization criteria.

    Args:
        values (pandas dataframe): dataframe of Binding Affinity values

    Returns:
        normalized_values (pandas dataframe): dataframe of values normalized between 0 and 1
    """    

def __compute_target__(pdb, targrp):
    """
    Calculates the case label from the binding affinity.
    Args:
    in_csv: path of the db1 csv
    """
    #in_csv = os.environ['TARGET_INPUT_CSV']
    in_csv="/projects/0/einf2380/data/external/processed/II/IDs_BA_DRB10101_MHCII_15mers.csv"
    df = pd.read_csv(in_csv)
    molname = targrp.parent.name.replace("/", "")
    class_id = normalize(df[df["ID"] == molname]["measurement_value"].values[0])
    print(np.array(class_id).shape)
    targrp.create_dataset("BA_REG", data=np.array(class_id))
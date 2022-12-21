from pdb2sql import pdb2sql
import pandas as pd
from deeprank.features import FeatureClass
from ast import literal_eval
import glob
import random
import pickle

class AnchorFeature(FeatureClass):

    def __init__(self, pdbfile):
        super().__init__("Residue")

        one_letter = {
            'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 
            'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',   
            'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
            'GLY':'G', 'PRO':'P', 'CYS':'C'
        }

        self.db = pdb2sql(pdbfile)
        self.df = pd.read_csv("/home/lepikhovd/3D-Vac/data/external/processed/all_hla.csv")
        self.peptide = [one_letter[res[1]] for res in self.db(chainID="P").get_residues()]
        anchors = self.df.loc[self.df["peptide"] == "".join(self.peptide)][["anchor_0", "anchor_1"]].values.tolist()[0]
        self.anchors = [str(anchor) for anchor in anchors]
        self.feature_data = {}
        self.feature_data_xyz = {}
    
    def get_feature(self):
        # get the feature's human readable format as well as xyz coordinates:
        # extract the keys and xyz of each key:      
        anch_keys = self.db.get("chainID, resSeq, resName", resSeq=self.anchors, chainID=["P"])
        print(anch_keys)
        anch_xyz = self.db.get("x,y,z", resSeq=self.anchors, chainID=["P"])

        # populate the dictionaries used to build the actual feature:
        hread, xyzval = {},{}

        for key, xyz in zip(anch_keys, anch_xyz):
            # print("anch_xyz:", anch_xyz)
            hread[tuple(key)] = [1.0]
            xyz_key = tuple([1] + xyz)
            xyzval[xyz_key] = [1.0]
            
        # assigning these two variables allows to build the features
        self.feature_data["anch"] = hread
        self.feature_data_xyz["anch"] = xyzval



def __compute_feature__(pdb_data, featgrp, featgrp_raw, chain1, chain2):

    # instantiate and build the feature:
    anchfeat = AnchorFeature(pdb_data)
    anchfeat.get_feature();

    # export to HDF5 format:
    anchfeat.export_dataxyz_hdf5(featgrp)
    anchfeat.export_data_hdf5(featgrp_raw)

    # close:
    anchfeat.db._close()

if __name__ == "__main__":
    # pdb_file = "/projects/0/einf2380/data/pMHCI/features_input_folder/test_case/pdb/BA-87169.pdb"
    df = pd.read_csv("../../data/external/processed/all_hla.csv")
    df["ID"] = df["ID"].apply(lambda x: x.replace("_","-"))

    csv_ids = df["ID"].tolist()
    all_models = pickle.load(open("../tools/all_models_paths.pkl", "rb"))
    all_models_ids = [model.split("/")[-1].replace(".pdb", "") for model in all_models]
    id_model_dict = pickle.load(open("./id_model_dict.pkl", "rb"))
    # id_model_dict = dict(zip(all_models_ids, all_models))
    
    # print("Filtering id_model_dict with csv cases..")
    # id_model_dict = {model_id: model for model_id, model in id_model_dict.items() if model_id in csv_ids}

    all_models_rand_ids = random.sample(list(id_model_dict.keys()), 10)

    for model_id in all_models_rand_ids:
        pdb_file = id_model_dict[model_id] 
        print(model_id)
        peptide = df.loc[df["ID"] == model_id, "peptide"].values[0]
        #create instance:
        anchfeat = AnchorFeature(pdb_file)
        anchfeat.get_feature()

        print(f"For case {model_id}:")
        print(f"peptide sequence: {peptide}")
        print(f"Anchors: {anchfeat.anchors}")
        print(f"Len of feature_data_xyz: {len(anchfeat.feature_data_xyz['anch'])}")
        print("#############################")
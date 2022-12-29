from pdb2sql import pdb2sql
from pdb2sql import interface
import pandas as pd
from deeprank.features import FeatureClass
from ast import literal_eval
import glob
import random
import pickle
import numpy as np

class AnchorFeature(FeatureClass):

    def __init__(self, pdbfile, contact_cutoff=5):
        super().__init__("Residue")

        one_letter = {
            'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 
            'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',   
            'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
            'GLY':'G', 'PRO':'P', 'CYS':'C'
        }

        self.contact_cutoff = contact_cutoff
        self.db = pdb2sql(pdbfile)
        self.idb = interface(self.db)
        self.df = pd.read_csv("/home/lepikhovd/3D-Vac/data/external/processed/all_hla.csv")
        self.peptide = [one_letter[res[1]] for res in self.db(chainID="P").get_residues()]
        anchors = self.df.loc[self.df["peptide"] == "".join(self.peptide)][["anchor_0", "anchor_1"]].values.tolist()[0]
        self.anchors = [str(anchor) for anchor in anchors]
        self.feature_data = {}
        self.feature_data_xyz = {}
    
    def get_feature(self):
        # get the feature's human readable format as well as xyz coordinates:
        # extract the keys and xyz of each key:      
        anch_p_keys = self.db.get("chainID, resSeq, resName", resSeq=self.anchors, chainID=["P"])
        anch_p_xyz = np.array(self.db.get("x,y,z", resSeq=self.anchors, chainID=["P"]))

        # those will be filled with m residues at the interface of anchors only:
        anch_m_keys = []
        anch_m_xyz = []

        # get the xyz value of each atom in chain M
        m_xyz = np.array(self.db.get("x,y,z", chainID=["M"]))

        # populate the dictionaries used to build the actual feature:
        hread, xyzval = {},{}

        # get all residues at the interface of both chains
        itf_residue_pairs = self.idb.get_contact_residues(
            cutoff=self.contact_cutoff,
            return_contact_pairs = True,
            chain1="P", 
            chain2="M")

        for (_, p_resSeq, p_resName), m_contact_residues in itf_residue_pairs.items():
            # if the residue number is an anchor then add the M chain residues info to anchor_feature:
            if str(p_resSeq) in self.anchors:
                for _, m_resSeq, __ in m_contact_residues:
                    anch_m_keys.extend(self.db.get("chainID, resSeq, resName", resSeq = m_resSeq, chainID = ["M"]))
                    anch_m_xyz.extend(self.db.get("x,y,z", resSeq = m_resSeq, chainID = ["M"]))
        
        # for test purposes only:
        # self.anch_m_keys = anch_m_keys
        # self.anch_m_xyz = anch_m_xyz
        # self.anch_p_keys = anch_p_keys
        # self.anch_p_xyz = anch_p_xyz

        # populate hread and xyzval with chain P:
        for key, xyz in zip(anch_p_keys, anch_p_xyz):
            hread[tuple(key)] = [1.0]
            xyz_key = tuple([1] + list(xyz))
            xyzval[xyz_key] = [1.0]

        # populate hread and xyzval with chain M:
        for key, xyz in zip(anch_m_keys, anch_m_xyz):
            hread[tuple(key)] = [0.0]
            xyz_key = tuple([0] + list(xyz))
            xyzval[xyz_key] = [0.0]
            
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

if __name__ == "__main__": #debuging purposes only
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
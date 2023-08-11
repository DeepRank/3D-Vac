from pdb2sql import pdb2sql
from pdb2sql import interface
import pandas as pd
from deeprank.features import FeatureClass
from ast import literal_eval
import glob
import random
import pickle
import numpy as np

class SequenceFeature(FeatureClass):

    def __init__(self, pdb_data):
        super().__init__("Residue")

        self.oneletter = {
            'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 
            'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',   
            'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
            'GLY':'G', 'PRO':'P', 'CYS':'C'
        }
        self.encoding = pd.read_csv('/projects/0/einf2380/data/external/processed/skip_gram_weights.tsv', delimiter='\t')
        pdb_db = pdb2sql(pdb_data)
        self.db = interface(pdb_db)
        self.feature_names = tuple([f'SkipGram_{n}' for n in self.encoding.index])
        
        for name in self.feature_names:
            self.feature_data[name] = {}
            self.feature_data_xyz[name] = {}
    
    def get_feature(self, chain1='M', chain2='P'):
        # get the feature's human readable format as well as xyz coordinates:
        # extract the keys and xyz of each key:      
        indA,indB = list(self.db.get_contact_atoms(chain1=chain1, 
                                                   chain2=chain2).values())
        contact = indA + indB
        # extract the atom keys and xyz of the contact CA atoms
        keys = self.db.get('serial,chainID,resName,resSeq,name',rowID=contact)
        xyz = self.db.get('x,y,z',rowID=contact)
        
        # populate the dictionaries used to build the actual feature:
        
        for name in self.feature_names:
            hread, xyzval = {},{}
            index = int(name.split('_')[1])
            # populate hread and xyzval with chain M:
            for key, coords in zip(keys, xyz):
                value = float(self.encoding[self.oneletter[key[2]]][index])
                # hread[tuple(key)] = [value]
                # xyz_key = tuple([0] + list(xyz))
                # xyzval[xyz_key] = [value]
                chain = [{chain1:0,chain2:1}[key[1]]]
                k = tuple(chain + coords)
                #Human readable
                hread[(key[1], key[3], 
                                   key[2], key[4])] = [value]
                #XYZ data
                xyzval[k] = [value]
                
            self.feature_data[name] = hread
            self.feature_data_xyz[name] = xyzval



def __compute_feature__(pdb_data, featgrp, featgrp_raw, chain1, chain2):
    # instantiate and build the feature:
    seqfeat = SequenceFeature(pdb_data)
    seqfeat.get_feature()

    # export to HDF5 format:
    seqfeat.export_dataxyz_hdf5(featgrp)
    seqfeat.export_data_hdf5(featgrp_raw)

    # close:
    seqfeat.db._close()

# if __name__ == "__main__": #debuging purposes only
#     df = pd.read_csv("../../data/external/processed/all_hla.csv")
#     df["ID"] = df["ID"].apply(lambda x: x.replace("_","-"))

#     csv_ids = df["ID"].tolist()
#     all_models = pickle.load(open("../tools/all_models_paths.pkl", "rb"))
#     all_models_ids = [model.split("/")[-1].replace(".pdb", "") for model in all_models]
#     id_model_dict = pickle.load(open("./id_model_dict.pkl", "rb"))
#     # id_model_dict = dict(zip(all_models_ids, all_models))
    
#     # print("Filtering id_model_dict with csv cases..")
#     # id_model_dict = {model_id: model for model_id, model in id_model_dict.items() if model_id in csv_ids}

#     all_models_rand_ids = random.sample(list(id_model_dict.keys()), 10)

#     for model_id in all_models_rand_ids:
#         pdb_file = id_model_dict[model_id] 
#         print(model_id)
#         peptide = df.loc[df["ID"] == model_id, "peptide"].values[0]
#         #create instance:
#         anchfeat = AnchorFeature(pdb_file)
#         anchfeat.get_feature()

#         print(f"For case {model_id}:")
#         print(f"peptide sequence: {peptide}")
#         print(f"Anchors: {anchfeat.anchors}")
#         print(f"Len of feature_data_xyz: {len(anchfeat.feature_data_xyz['anch'])}")
#         print("#############################")
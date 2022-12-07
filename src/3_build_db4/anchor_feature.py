from pdb2sql import pdb2sql
import pandas as pd
from deeprank.features import FeatureClass
from ast import literal_eval

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
        self.df = pd.read_csv("/home/lepikhovd/3D-Vac/data/external/processed/test_case.csv")
        self.peptide = [one_letter[res[1]] for res in self.db(chainID="P").get_residues()]
        anchors = literal_eval(self.df[self.df["peptide"] == "".join(self.peptide)]["anchors"][0])
        self.anchors = [str(a) for a in anchors]
        self.feature_data = {}
        self.feature_data_xyz = {}
    
    def get_feature(self):
        # get the feature's human readable format as well as xyz coordinates:
        # extract the keys and xyz of each key:      
        anch_keys = self.db.get("chainID, resSeq, resName", resSeq=self.anchors, chainID=["P"])
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
    pdb_file = "/projects/0/einf2380/data/pMHCI/features_input_folder/test_case/pdb/BA_76958.pdb"

    #create instance:
    anchfeat = AnchorFeature(pdb_file)

    anchfeat.get_feature()
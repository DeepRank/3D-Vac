from pdb2sql import pdb2sql
import pandas as pd
from deeprank.features import FeatureClass

class AnchorFeature(FeatureClass):

    def __init__(self, pdbfile):
        super().__init__("Residue")

        one_letter = {
            'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', 
            'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',   
            'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
            'GLY':'G', 'PRO':'P', 'CYS':'C'
        }

        self.pdb = pdbfile
        self.db = pdb2sql(self.pdb)
        self.df = pd.read_csv("/home/lepikhovd/3D-Vac/data/external/processed/test_case.csv")
        self.peptide = [one_letter[res[1]] for res in self.db(chainID="P").get_residues()]
        self.anchors = self.df[self.df["peptide"] == "".join(self.peptide)]["anchors"]
    
    def get_feature(self):
        print("self anchors:", self.anchors)

def __compute_feature__(pdb_data, featgrp, featgrp_raw, chain1, chain2):

    anchfeat = AnchorFeature(pdb_data)
    anchfeat.get_feature();
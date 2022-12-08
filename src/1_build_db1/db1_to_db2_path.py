# -*- coding: utf-8 -*-
# Script used to add the destination column for the PANDORA wrapper. When the DB1 is used to build DB2,
# the column is used as the destination path for the folder containing the pdb structures.
# For now, this file shoud be used as a script for DB1_II and a module for DB1_I.
import pickle

def assign_outfolder(index):

    if index%1000 != 0:
        interval = '%i_%i' %((int(index/1000)*1000)+1, (int(index/1000)*1000)+1000)
    else:
        interval = '%i_%i' %(index-999, index)
    return interval

if __name__ == "__main__":
    #%%
    with open('../binding_data/pMHCII/ba_values.pkl', 'rb') as inpkl:
        ba_values = pickle.load(inpkl)
        ba_dict = pickle.load(inpkl)

    MHCII_outfolder = '/projects/0/einf2380/data/pMHCII/3D_models/'
    BA_outfolder = MHCII_outfolder + 'BA/'
    EL_outfolder = MHCII_outfolder + 'EL/'

    #%%
    with open('../binding_data/pMHCII/IDs_BA_MHCII.csv', 'w') as outcsv:
        with open('../binding_data/pMHCII/IDs_BA_DRB0101_MHCII.csv', 'w') as studycase_csv:
            for i, case in enumerate(ba_values):
                index = i + 1
                ID = 'BA_%i' %index
                pept = case[0]
                allele = case[1]
                score = str(case[2])
                ba = str(case[3])
                out_interval = assign_outfolder(index)
                outfolder = BA_outfolder + out_interval
                row = [ID, pept, allele, score, ba, outfolder + '\n']
                outcsv.write((',').join(row))
                if allele == 'DRB1_0101':
                    studycase_csv.write((',').join(row))

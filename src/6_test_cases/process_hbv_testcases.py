import sys
sys.path.append('../5_train_models/str/PyTorch/')
from process_hdf5 import *
import glob
import torch


def parse_pdb_dataset(folder, return_tuple=False, one_hot=False, residue_level=True, radius_pocket=10, elements=['C', 'N', 'O', 'S', 'P'], exclude_elements=['H']):
    data_list = []
            
    for pdb_path in glob.glob(folder):
        
        data = parse_pdb_entry(pdb_path, return_tuple=return_tuple, one_hot=one_hot, residue_level=residue_level, radius_pocket=radius_pocket, elements=elements, exclude_elements=exclude_elements)
        data_list.append(data)

    return data_list

def parse_pdb_entry(pdb_path, return_tuple=False, one_hot=False, residue_level=True, radius_pocket=10, elements=['C', 'N', 'O', 'S', 'P'], exclude_elements=['H']):
    pdb_id = pdb_path.split('/')[-1].split('.')[0]
    
    with open(pdb_path, 'r') as infile:
        pdbf = []
        for line in infile.readlines():
            pdbf.append([x for x in line.replace('\n','').split(' ') if x != ''])
            
    data_M, data_P = parse_h5py_pMHC_complex(pdbf, hdf5=False, one_hot=one_hot, residue_level=residue_level, radius_pocket=radius_pocket, elements=elements, exclude_elements=exclude_elements)

    target_bin = torch.LongTensor([1.0])
    #target_reg = torch.FloatTensor([1.0])
    target_reg = torch.LongTensor([1.0])

    if return_tuple:
        data_M.target_bin = target_bin
        data_M.target_reg = target_reg
        data_P.target_bin = target_bin
        data_P.target_reg = target_reg

        data_M.name = pdb_id
        data_P.name = pdb_id

        return data_M, data_P
    else:
        # entity 0 for M 1 for P
        data = (
            Data(x=torch.cat([data_M.x, data_P.x], dim=0), 
                 pos=torch.cat([data_M.pos, data_P.pos], dim=0), 
                 mass=torch.cat([data_M.mass, data_P.mass], dim=0) if not residue_level else None,
                 id=pdb_id,
                 y_bin=target_bin,
                 y_reg=target_reg,
                 entity = torch.cat([torch.zeros(data_M.x.shape[0]), torch.ones(data_P.x.shape[0])], dim=0)
                )
        )

    return data

data = parse_pdb_dataset('/home/dmarz/test_cases/final_folders/hbv_testcase/*')
torch.save(data, '/home/dmarz/test_cases/final_folders/egnn_test_dataset.pt')


# db = torch.load('/projects/0/einf2380/data/pMHCI/egnn_data/supervised/pandora_dataset.pt')
# db_full = db + data
# torch.save(db_full, '/home/dmarz/test_cases/final_folders/egnn_extended_full_dataset.pt')
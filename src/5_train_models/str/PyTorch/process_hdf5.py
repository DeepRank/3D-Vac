"""
This file is the code to parse the h5py PANDORA dataset.

parse_h5py_pMHC_dataset will process one h5py file into a list of torch_geometric.data.Data objects

To replicate my data extraction, you can use the following code:

data_list = []

for file in hdf5_files:
    data_list += parse_h5py_pMHC_dataset(file, return_tuple=False, one_hot=False, residue_level=True, radius_pocket=10, elements=['C', 'N', 'O', 'S', 'P'], exclude_elements=['H'])

torch.save(data_list, 'pandora_pMHC_residue_level.pt')


Then later to load the data:

data_list = torch.load('pandora_pMHC_residue_level.pt')

loader = DataLoader(data_list, batch_size=1, shuffle=True) etc. 
"""


from .utils import filter_element_dictionaries, res_ids2idx

import os
import re
from collections import defaultdict

import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius

import h5py

elements = [
    'H', 'HE', 'LI', 'BE', 'B', 'C', 'N', 'O', 'F', 'NE', 'NA', 'MG', 'AL', 'SI', 'P', 'S', 'CL', 'AR', 'K', 'CA', 'SC',
    'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'GE', 'AS', 'SE', 'BR', 'KR', 'RB', 'SR', 'Y', 'ZR', 'NB',
    'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'SB', 'TE', 'I', 'XE', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'PM',
    'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL',
    'PB', 'BI', 'PO', 'AT', 'RN', 'FR', 'RA', 'AC', 'TH', 'PA', 'U', 'NP', 'PU', 'AM', 'CM', 'BK', 'CF', 'ES', 'FM', 'MD',
    'NO', 'LR', 'RF', 'DB', 'SG', 'BH', 'HS', 'MT', 'DS', 'RG', 'CN', 'NH', 'FL', 'MC', 'LV', 'TS', 'OG'
]

el2mass = {
    'H': 1.00794, 'HE': 4.002602, 'LI': 6.941, 'BE': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 
    'F': 18.9984032, 'NE': 20.1797, 'NA': 22.98976928, 'MG': 24.305, 'AL': 26.9815386, 'SI': 28.0855, 'P': 30.973762, 
    'S': 32.065, 'CL': 35.453, 'AR': 39.948, 'K': 39.0983, 'CA': 40.078, 'SC': 44.955912, 'TI': 47.867, 'V': 50.9415, 
    'CR': 51.9961, 'MN': 54.938045, 'FE': 55.845, 'CO': 58.933195, 'NI': 58.6934, 'CU': 63.546, 'ZN': 65.409, 'GA': 69.723, 
    'GE': 72.64, 'AS': 74.9216, 'SE': 78.96, 'BR': 79.904, 'KR': 83.798, 'RB': 85.4678, 'SR': 87.62, 'Y': 88.90585, 'ZR': 91.224, 
    'NB': 92.90638, 'MO': 95.94, 'TC': 98.9063, 'RU': 101.07, 'RH': 102.9055, 'PD': 106.42, 'AG': 107.8682, 'CD': 112.411, 
    'IN': 114.818, 'SN': 118.71, 'SB': 121.76, 'TE': 127.6, 'I': 126.90447, 'XE': 131.293, 'CS': 132.9054519, 'BA': 137.327, 
    'LA': 138.90547, 'CE': 140.116, 'PR': 140.90465, 'ND': 144.242, 'PM': 146.9151, 'SM': 150.36, 'EU': 151.964, 'GD': 157.25, 
    'TB': 158.92535, 'DY': 162.5, 'HO': 164.93032, 'ER': 167.259, 'TM': 168.93421, 'YB': 173.04, 'LU': 174.967, 'HF': 178.49, 
    'TA': 180.9479, 'W': 183.84, 'RE': 186.207, 'OS': 190.23, 'IR': 192.217, 'PT': 195.084, 'AU': 196.966569, 'HG': 200.59, 
    'TL': 204.3833, 'PB': 207.2, 'BI': 208.9804, 'PO': 208.9824, 'AT': 209.9871, 'RN': 222.0176, 'FR': 223.0197, 'RA': 226.0254, 
    'AC': 227.0278, 'TH': 232.03806, 'PA': 231.03588, 'U': 238.02891, 'NP': 237.0482, 'PU': 244.0642, 'AM': 243.0614, 
    'CM': 247.0703, 'BK': 247.0703, 'CF': 251.0796, 'ES': 252.0829, 'FM': 257.0951, 'MD': 258.0951, 'NO': 259.1009, 
    'LR': 262, 'RF': 267, 'DB': 268, 'SG': 271, 'BH': 270, 'HS': 269, 'MT': 278, 'DS': 281, 'RG': 281, 'CN': 285, 
    'NH': 284, 'FL': 289, 'MC': 289, 'LV': 292, 'TS': 294, 'OG': 294
}

el2idx = {el: i for i, el in enumerate([key for key in el2mass.keys()])}
idx2el = {i: el for i, el in enumerate([key for key in el2mass.keys()])}

idx2mass = {i: el2mass[el] for i, el in enumerate([key for key in el2mass.keys()])}

res_ids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'SEC', 'GLX', 'UNK']
res_ids2idx = {res_name: i for i, res_name in enumerate(res_ids)}


def filter_element_dictionaries(elements=None, exclude_elements=None):
    """Filter the dictionaries el2mass and el2idx to only contain the elements in the list elements."""

    global el2mass, el2idx

    if elements is None and exclude_elements is None:
        return el2idx, el2mass
    elif elements is not None:
        _el2mass = {el: mass for el, mass in el2mass.items() if el in elements}
        _el2idx = {el: i for el, i in el2idx.items() if el in elements}

        _el2idx = {el: i for i, el in enumerate([key for key in _el2mass.keys()])}
    elif exclude_elements is not None:
        _el2mass = {el: mass for el, mass in el2mass.items() if el not in exclude_elements}
        _el2idx = {el: i for el, i in el2idx.items() if el not in exclude_elements}

        _el2idx = {el: i for i, el in enumerate([key for key in _el2mass.keys()])}

    return _el2idx, _el2mass


def parse_h5py_pMHC_complex(h5py_pdb, one_hot=True, residue_level=False, radius_pocket=10, elements=['C', 'N', 'O', 'S', 'P'], exclude_elements=['H']):
    el2idx, el2mass = filter_element_dictionaries(elements=elements, exclude_elements=exclude_elements)

    pos_M, pos_P, x_M, x_P, mass_M, mass_P, res_M, res_P = None, None, None, None, None, None, None, None

    for line in h5py_pdb:
        line = line.decode('utf-8').split()

        if line[0] != 'ATOM':
            continue

        atom_type = line[2].upper() # type of atom within the residue (CA, CB, etc.)
        element = line[-1].upper() # atom element

        res_ids = line[3]
        x, y, z = float(line[6]), float(line[7]), float(line[8])

        if residue_level:
            if atom_type == 'CA':
                if line[4] == 'M':
                    if pos_M is None:
                        pos_M = []
                        x_M = []
                        res_M = []
                    pos_M.append([x, y, z])
                    if res_ids not in res_ids2idx:
                        if res_ids[0] == 'A':
                            res_ids = res_ids[1:]
                        else:
                            continue
                    x_M.append(res_ids2idx[res_ids])
                    res_M.append(res_ids)
                elif line[4] == 'P':
                    if pos_P is None:
                        pos_P = []
                        x_P = []
                        res_P = []
                    pos_P.append([x, y, z])
                    x_P.append(res_ids2idx[res_ids])
                    res_P.append(res_ids)
        else:
            if line[4] == 'M':
                if pos_M is None:
                    pos_M = []
                    x_M = []
                    mass_M = []
                pos_M.append([x, y, z])
                x_M.append(el2idx[element])
                mass_M.append(el2mass[element])
            elif line[4] == 'P':
                if pos_P is None:
                    pos_P = []
                    x_P = []
                    mass_P = []
                pos_P.append([x, y, z])
                x_P.append(el2idx[element])
                mass_P.append(el2mass[element])

    if residue_level:
        if pos_M is not None:
            pos_M = torch.tensor(pos_M)
            x_M = torch.LongTensor(x_M)
            if one_hot:
                x_M = torch.nn.functional.one_hot(x_M, num_classes=len(res_ids2idx)).float()
            data_M = Data(x=x_M, pos=pos_M)
        else:
            data_M = None

        if pos_P is not None:
            pos_P = torch.tensor(pos_P)
            x_P = torch.LongTensor(x_P)
            if one_hot:
                x_P = torch.nn.functional.one_hot(x_P, num_classes=len(res_ids2idx)).float()
            data_P = Data(x=x_P, pos=pos_P)
        else:
            data_P = None
    else:
        if pos_M is not None:
            pos_M = torch.tensor(pos_M)
            x_M = torch.LongTensor(x_M)
            mass_M = torch.FloatTensor(mass_M)
            if one_hot:
                x_M = torch.nn.functional.one_hot(x_M, num_classes=len(el2idx)).float()
            data_M = Data(x=x_M, pos=pos_M, mass=mass_M)
        else:
            data_M = None

        if pos_P is not None:
            pos_P = torch.tensor(pos_P)
            x_P = torch.LongTensor(x_P)
            mass_P = torch.FloatTensor(mass_P)
            if one_hot:
                x_P = torch.nn.functional.one_hot(x_P, num_classes=len(el2idx)).float()
            data_P = Data(x=x_P, pos=pos_P, mass=mass_P)
        else:
            data_P = None
    
    if radius_pocket is not None:
        assign_index = radius(data_M.pos, data_P.pos, radius_pocket, max_num_neighbors=10000)
        pocket_indices = torch.unique(assign_index[1])
        
        data_M.x = data_M.x[pocket_indices]
        data_M.pos = data_M.pos[pocket_indices]

        if not residue_level:
            data_M.mass = data_M.mass[pocket_indices]

    return data_M, data_P


def parse_h5py_pMHC_entry(h5py_entry, return_tuple=False, one_hot=True, residue_level=False, radius_pocket=10, elements=['C', 'N', 'O', 'S', 'P'], exclude_elements=['H']):
    data_M, data_P = parse_h5py_pMHC_complex(h5py_entry['complex'], one_hot=one_hot, residue_level=residue_level, radius_pocket=radius_pocket, elements=elements, exclude_elements=exclude_elements)

    target_bin = torch.LongTensor([h5py_entry['targets']['BIN_CLASS'][()]])
    target_reg = torch.FloatTensor([h5py_entry['targets']['CONTINUOUS_NORM'][()]])

    if return_tuple:
        data_M.target_bin = target_bin
        data_M.target_reg = target_reg
        data_P.target_bin = target_bin
        data_P.target_reg = target_reg

        data_M.name = h5py_entry.name
        data_P.name = h5py_entry.name

        return data_M, data_P
    else:
        # entity 0 for M 1 for P
        data = (
            Data(x=torch.cat([data_M.x, data_P.x], dim=0), 
                 pos=torch.cat([data_M.pos, data_P.pos], dim=0), 
                 mass=torch.cat([data_M.mass, data_P.mass], dim=0) if not residue_level else None,
                 id=h5py_entry.name,
                 target_bin=target_bin,
                 target_reg=target_reg,
                 entity = torch.cat([torch.zeros(data_M.x.shape[0]), torch.ones(data_P.x.shape[0])], dim=0)
                )
        )

    return data


def parse_h5py_pMHC_dataset(h5py_file, return_tuple=False, one_hot=True, residue_level=False, radius_pocket=10, elements=['C', 'N', 'O', 'S', 'P'], exclude_elements=['H']):
    data_list = []

    h5py_file = h5py.File(h5py_file, 'r')

    for i, h5py_id in enumerate(h5py_file):
        if i % 500 == 0:
            print(f'Processed {i*100}/{len(h5py_file)}% entries')
            
        h5py_entry = h5py_file[h5py_id]
        data = parse_h5py_pMHC_entry(h5py_entry, return_tuple=return_tuple, one_hot=one_hot, residue_level=residue_level, radius_pocket=radius_pocket, elements=elements, exclude_elements=exclude_elements)
        data_list.append(data)

    return data_list
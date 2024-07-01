import torch
from torch_geometric.loader import DataLoader

import sys
sys.path.append('../5_train_models/str/PyTorch/')
from egnn import EGNNModel
from data_proccess_fn import data_process_fn

exclude_keys_all = [
    "angle_index",
    "angle_targets",
    "dihedral_index",
    "dihedral_targets",
    "residue_index",
    "residue_id",
]

data = torch.load('/home/dmarz/test_cases/final_folders/egnn_test_dataset.pt')

loader = DataLoader(
    data, batch_size=2, shuffle=False, exclude_keys=exclude_keys_all
)

gnn = EGNNModel(
    num_layers=3,
    emb_dim=128,
    edge_dim=1,
    in_dim=23,
    out_dim=1,
    pool="peptide_sum",
    cond_method="strong",
    deep_conditioning=False,
    separate_entity_embeddings=False,
    embed=True,
    rbf=True,
    rbf_max=30,
    rbf_dim=64,
    update_pos=False,
    dropout=0.1,
)

gnn.load_state_dict(torch.load('/projects/0/einf2380/data/pMHCI/trained_models/EGNN/HBV_testcase_best_model.pt', map_location=torch.device('cpu')))

gnn.eval()

outputs = []
for x in loader:
    x = data_process_fn(x)
    outputs.append(gnn(x))

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

print(f'OUTPUT: {sigmoid(outputs[0].detach().numpy())}')
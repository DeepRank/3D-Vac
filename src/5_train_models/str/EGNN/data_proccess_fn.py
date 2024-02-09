from functools import partial

import torch
from torch_geometric.transforms import KNNGraph

knn_graph = KNNGraph(k=10, loop=False, force_undirected=True)

def add_edge_and_cross_entity_edge_type(data, edge_transform=knn_graph):
    data = edge_transform(data)
    data.edge_attr = torch.zeros(data.edge_index.shape[1], dtype=torch.long).unsqueeze(-1).to(data.edge_index.device)

    mask_different = data.entity[data.edge_index[0]] != data.entity[data.edge_index[1]]
    data.edge_attr[mask_different] = 1

    return data

data_process_fn = partial(add_edge_and_cross_entity_edge_type, edge_transform=knn_graph)
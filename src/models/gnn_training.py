import torch.nn as nn
import h5py
import torch
from deeprank_gnn.DataSet import HDF5DataSet
from deeprank_gnn.NeuralNet import NeuralNet
from gnn_architecture import Binding_GNN
import numpy as np
from deeprank_gnn.ginet import GINet

# Number of cores the program is using
torch.set_num_threads(20)

##############################
#         CONSTANTS          #
##############################
MESSAGE_VECTOR_LENGTH   = 32 #
NMB_HIDDED_ATTRIBUTES   = 32 #
NMB_EDGE_PROJECTION     = 32 #
NMB_OUPUT_FEATURES      =  2 #
NMB_MLP_NEURONS         = 32 #
NMB_GNN_LAYERS          =  3 #
##############################

database = '/Users/giuliacrocioni/remote_snellius/data/pMHCI/gnn_hdf5/rd.hdf5'
node_features = ["bsa", "depth", "hb_acceptors", "hb_donors", "hse", "ic", "pssm", "polarity", "sasa", "size", "type"]
edge_features = ["coulomb", "covalent", "dist", "vanderwaals"]

if __name__ == '__main__':

    # creating HDF5DataSet object and selecting molecules, node/edge features, target, and clustering method:
    dataset = HDF5DataSet(
        root = "./",
        database = database,
        node_feature = node_features,
        edge_feature = edge_features,
        target = "binary",
        clustering_method = None
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ############### TO BE REMOVED FROM HERE ###############
    # def get_edge_values(hdf5, sample_key, features = ['coulomb', 'covalent', 'dist', 'vanderwaals']):
    #     edge_keys = hdf5[sample_key]['edge_data'].keys()
    #     edge_values = [torch.tensor(np.array( hdf5[sample_key]['edge_data'][key]))
    #                     for key in features]
    #     for vi in range(len(edge_values)):
    #         if len(edge_values[vi].shape) == 1:
    #             edge_values[vi] = edge_values[vi].unsqueeze(1)
    #     edge_values = torch.cat(edge_values,1)
    #     edge_index = torch.tensor(np.array(hdf5[sample_key]['edge_index']))
    #     edge_values = edge_values[edge_index[:,0] != edge_index[:,1]]
    #     return torch.nan_to_num(edge_values.float())

    # def get_node_values(hdf5, sample_key, features=['bsa', 'depth', 'hb_acceptors', 'hb_donors', 'hse', 'ic', 'polarity', 'pos', 'pssm', 'sasa', 'size', 'type']):
    #     node_keys = hdf5[sample_key]['node_data'].keys()
    #     node_values = [torch.tensor(np.array( hdf5[sample_key]['node_data'][key]))
    #                     for key in features]
    #     # Add features that node is peptide or mhc:
    #     #node_values += [torch.tensor([int(str(i).split(' ')[-2] == 'P') 
    #     #						 for i in list(hdf5[sample_key]['nodes'])])]
    #     for vi in range(len(node_values)):
    #         if len(node_values[vi].shape) == 1:
    #             node_values[vi] = node_values[vi].unsqueeze(1)
    #     node_values = torch.cat(node_values,1)
    #     return torch.nan_to_num(node_values.float())

    # with h5py.File(database, 'r') as hdf5:
    #     samples = [i for i in hdf5.keys()]
    
    #     # Calculate how many inputs the node and edge features have
    #     NMB_EDGE_ATTIBUTES  = get_edge_values(hdf5, samples[0], edge_features).shape[1]
    #     NMB_NODE_ATTRIBUTES = get_node_values(hdf5, samples[0], node_features).shape[1]
    ############### TO BE REMOVED FROM HERE ###############

    # # Init GNN
    # gnn = Binding_GNN(nmb_edge_attr=NMB_EDGE_ATTIBUTES, nmb_node_attr=NMB_NODE_ATTRIBUTES, 
    #             nmb_output_features=NMB_OUPUT_FEATURES, nmb_gnn_layers=NMB_GNN_LAYERS,
    #             message_vector_length=MESSAGE_VECTOR_LENGTH, nmb_mlp_neurons=NMB_MLP_NEURONS,
    #             nmb_hidden_attr=NMB_HIDDED_ATTRIBUTES, nmb_edge_projection=NMB_EDGE_PROJECTION)

    nn = NeuralNet(
        dataset,
        GINet,
        task = "class",
        batch_size = 16,
        percent = [0.8, 0.2]
    )

    # nn.optimizer = torch.optim.Adam(gnn.parameters(), weight_decay=1e-5)
    # nn.loss = nn.CrossEntropyLoss()

    nn.train(nepoch = 5, validate = True)
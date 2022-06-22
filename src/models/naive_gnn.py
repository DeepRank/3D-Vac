import logging
import torch
from torch.nn import Parameter, Module, Linear, Sequential, ReLU

from torch_scatter import scatter_mean, scatter_sum

class NaiveConvolutionalLayer(Module):
    def __init__(self, count_node_features, count_edge_features):
        super(NaiveConvolutionalLayer, self).__init__()
        message_size = 32
        hidden_size = 64
        edge_input_size = 2 * count_node_features + count_edge_features
        self._edge_mlp = Sequential(Linear(edge_input_size, message_size), ReLU())
        node_input_size = count_node_features + message_size
        self._node_mlp = Sequential(Linear(node_input_size, count_node_features), ReLU())
    def forward(self, node_features, edge_node_indices, edge_features):
        # generate messages over edges
        node0_indices, node1_indices = edge_node_indices
        node0_features = node_features[node0_indices]
        node1_features = node_features[node1_indices]
        message_input = torch.cat([node0_features, node1_features, edge_features], dim=1)
        messages_per_neighbour = self._edge_mlp(message_input)
        # aggregate messages
        out = torch.zeros(node_features.shape[0], messages_per_neighbour.shape[1]).to(node_features.device)
        message_sums_per_node = scatter_sum(messages_per_neighbour, node0_indices, dim=0, out=out)
        # update nodes
        node_input = torch.cat([node_features, message_sums_per_node], dim=1)
        node_output = self._node_mlp(node_input)
        return node_output
class NaiveNetwork(Module):
    def __init__(self, input_shape, output_shape, input_shape_edge):
        """
            Args:
                input_shape(int): number of node input features
                output_shape(int): number of output value per graph
                input_shape_edge(int): number of edge input features
        """
        super(NaiveNetwork, self).__init__()
#        self._internal1 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
#        self._internal2 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        self._external1 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        self._external2 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        hidden_size = 128
        self._graph_mlp = Sequential(Linear(input_shape, hidden_size), ReLU(), Linear(hidden_size, output_shape))
    def forward(self, data):
#        internal_updated1_node_features = self._internal1(data.x, data.internal_edge_index, data.internal_edge_attr)
#        internal_updated2_node_features = self._internal2(internal_updated1_node_features, data.internal_edge_index, data.internal_edge_attr)
        external_updated1_node_features = self._external1(data.x, data.edge_index, data.edge_attr)
        external_updated2_node_features = self._external2(external_updated1_node_features, data.edge_index, data.edge_attr)
#        means_per_graph_internal = scatter_mean(internal_updated2_node_features, data.batch, dim=0)
        means_per_graph_external = scatter_mean(external_updated2_node_features, data.batch, dim=0)
#        interest_internal = internal_updated2_node_features[data.node_of_interest_index]
#        interest_external = external_updated2_node_features[data.node_of_interest_index]
        #graph_input = torch.cat([interest_internal, interest_external], dim=1)
        graph_input = means_per_graph_external
        z = self._graph_mlp(graph_input)
        return z
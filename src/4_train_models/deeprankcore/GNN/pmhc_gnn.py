import torch
from torch.nn import Linear, Module, ReLU, Sequential, BatchNorm1d, LayerNorm
from torch_scatter import scatter_mean, scatter_sum
import torch_geometric


class NaiveConvolutionalLayer(Module):

    def __init__(self, count_node_features, count_edge_features):
        super().__init__()
        message_size = 32
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


class Conv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_dim, out_dim, attr_dim, bias=True, aggr="add", depthwise=False):
        super().__init__(node_dim=0, aggr=aggr)

        self.in_dim, self.out_dim = in_dim, out_dim
        self.depthwise = depthwise

        if self.depthwise:
            self.einsum_eq = '...o,...o->...o'
            if in_dim != out_dim:
                raise ValueError(f"When depthwise=True in- and output dimensions should be the same")
            self.kernel = torch.nn.Linear(attr_dim, out_dim, bias=False)
        else:
            self.einsum_eq = '...oi,...i->...o'
            self.kernel = torch.nn.Linear(attr_dim, out_dim * in_dim, bias=False)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_dim))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_attr, edge_index, batch_target=None, batch_source=None):
        if batch_source is not None and batch_target is not None:
            size = (batch_source.shape[0], batch_target.shape[0])
        else:
            size = None
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size = size)

    def message(self, x_j, edge_attr):
        kernel = self.kernel(edge_attr)
        if not self.depthwise:
            kernel = kernel.unflatten(-1, (self.out_dim, self.in_dim))
        return torch.einsum(self.einsum_eq, kernel, x_j) * 1.
    
    def update(self, message_aggr):
        out = message_aggr
        if self.bias is not None:
            out += self.bias
        return out


class ConvNextLayer(torch.nn.Module):
    def __init__(self, feature_dim, attr_dim, act=torch.nn.ReLU(), layer_scale=1e-6,aggr="add"): 
        super().__init__()

        self.conv = Conv(feature_dim, feature_dim, attr_dim, aggr=aggr, depthwise=True)
        self.act_fn = act
        self.linear_1 = torch.nn.Linear(feature_dim, 4 * feature_dim)
        self.linear_2 = torch.nn.Linear(4 * feature_dim, feature_dim)
        self.layer_scale = torch.nn.Parameter(torch.ones(1, feature_dim) * layer_scale)
        self.norm = LayerNorm(feature_dim)

    def forward(self, x, attr, edge_index, batch, batch_source=None):
        input = x
        x = self.conv(x, attr, edge_index, batch_target=batch, batch_source=batch_source)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        x = self.layer_scale * x
        if input.shape == x.shape: 
            x = x + input
        return x


class PMHCI_Network01(Module):

    def __init__(self, input_shape: int, output_shape: int, input_shape_edge: int):
        """
        First improved naive graph neural network.
            Added batch normalization in the last mlp layers, and increased number of 
            parameters by adding a convolutional layer and a dense layer. 
    
        Args:
            input_shape (int): Number of node input features.
            output_shape (int): Number of output value per graph.
            input_shape_edge (int): Number of edge input features.
        """
        super().__init__()
        self._external1 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        self._external2 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        self._external3 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        hidden_size = 128
        self._graph_mlp = Sequential(
            Linear(input_shape, hidden_size), BatchNorm1d(hidden_size), ReLU(),
            Linear(hidden_size, hidden_size), BatchNorm1d(hidden_size), ReLU(),
            Linear(hidden_size, output_shape))

    def forward(self, data):
        external_updated1_node_features = self._external1(data.x, data.edge_index, data.edge_attr)
        external_updated2_node_features = self._external2(external_updated1_node_features, data.edge_index, data.edge_attr)
        external_updated3_node_features = self._external3(external_updated2_node_features, data.edge_index, data.edge_attr)
        means_per_graph_external = scatter_mean(external_updated3_node_features, data.batch, dim=0)
        graph_input = means_per_graph_external
        z = self._graph_mlp(graph_input)
        return z


class PMHCI_Network02(Module):

    def __init__(self, input_shape: int, output_shape: int, input_shape_edge: int):
        """
        Args:
            input_shape (int): Number of node input features.
            output_shape (int): Number of output value per graph.
            input_shape_edge (int): Number of edge input features.
        """
        super().__init__()
        self._external1 = ConvNextLayer(input_shape, input_shape_edge)
        self._external2 = ConvNextLayer(input_shape, input_shape_edge)
        hidden_size = 128
        self._graph_mlp = Sequential(Linear(input_shape, hidden_size), ReLU(), Linear(hidden_size, output_shape))

    def forward(self, data):
        external_updated1_node_features = self._external1(data.x, data.edge_attr, data.edge_index, data.batch)
        external_updated2_node_features = self._external2(external_updated1_node_features, data.edge_attr, data.edge_index, data.batch)
        means_per_graph_external = scatter_mean(external_updated2_node_features, data.batch, dim=0)
        graph_input = means_per_graph_external
        z = self._graph_mlp(graph_input)
        return z


class PMHCI_Network03(Module):

    def __init__(self, input_shape: int, output_shape: int, input_shape_edge: int):
        """
        First improved naive graph neural network.
            Added batch normalization in the last mlp layers, and increased number of 
            parameters by adding a convolutional layer and a dense layer. 
    
        Args:
            input_shape (int): Number of node input features.
            output_shape (int): Number of output value per graph.
            input_shape_edge (int): Number of edge input features.
        """
        super().__init__()
        self._external1 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        self._external2 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        hidden_size = 128
        self._graph_mlp = Sequential(
            Linear(input_shape, hidden_size), ReLU(),
            Linear(hidden_size, hidden_size), ReLU(),
            Linear(hidden_size, output_shape))

    def forward(self, data):
        external_updated1_node_features = self._external1(data.x, data.edge_index, data.edge_attr)
        external_updated2_node_features = self._external2(external_updated1_node_features, data.edge_index, data.edge_attr)
        means_per_graph_external = scatter_mean(external_updated2_node_features, data.batch, dim=0)
        graph_input = means_per_graph_external
        z = self._graph_mlp(graph_input)
        return z
    
class NaiveGNN1(Module):

    def __init__(self, input_shape: int, output_shape: int, input_shape_edge: int):
        """
        Args:
            input_shape (int): Number of node input features.
            output_shape (int): Number of output value per graph.
            input_shape_edge (int): Number of edge input features.
        """
        super().__init__()
        self._external1 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        self._external2 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        self._external3 = NaiveConvolutionalLayer(input_shape, input_shape_edge)
        hidden_size = 128
        self._graph_mlp = Sequential(
            Linear(input_shape, hidden_size), ReLU(),
            Linear(hidden_size, hidden_size), ReLU(),
            Linear(hidden_size, output_shape))

    def forward(self, data):
        external_updated1_node_features = self._external1(data.x, data.edge_index, data.edge_attr)
        external_updated2_node_features = self._external2(external_updated1_node_features, data.edge_index, data.edge_attr)
        external_updated3_node_features = self._external3(external_updated2_node_features, data.edge_index, data.edge_attr)
        means_per_graph_external = scatter_mean(external_updated3_node_features, data.batch, dim=0)
        graph_input = means_per_graph_external
        z = self._graph_mlp(graph_input)
        return z

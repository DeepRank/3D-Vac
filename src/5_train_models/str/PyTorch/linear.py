import torch
import torch.nn as nn

from torch_geometric.nn.pool import global_mean_pool, global_add_pool

class ConditionalLinear(nn.Module):
    """
    This is a conditional linear layer.

    You can use it to conditionally transform a feature vector, in our case a distance between two atoms.

    In the 'weak' regime, it just concatenates the conditional vector to the feature vector and applies a linear transformation.
    In the 'strong' regime, it applies a linear transformation to the feature vector and then multiplies it with a linear transformation of the conditional vector.
    In the 'pure' regime, it applies a bilinear transformation to the feature vector and the conditional vector.
    """


    def __init__(self,
                 in_features,
                 out_features,
                 cond_features,
                 method='weak'):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cond_features = cond_features
        self.method = method

        # Using torch nn Linear and BiLinear
        # The weights that parametrize the conditional linear layer
        if method == 'weak':
            self.linear_weak = nn.Linear(self.in_features + self.cond_features, self.out_features)
        elif method == 'strong':
            self.linear = nn.Linear(self.in_features, self.out_features)
            self.linear_embedding = nn.Linear(self.cond_features, self.out_features, bias=False)
        elif method == 'pure':
            self.bilinear = nn.Bilinear(self.in_features, self.cond_features, self.out_features)
        else:
            raise ValueError('Unknown method, should be \'weak\', \'strong\', or \'pure\'.')

    def forward(self, f_in, cond_vec, add_vec=None, add_index=None):
        if self.method == 'weak':
            f_out = self.linear_weak(torch.cat((f_in, cond_vec), dim=-1))
        elif self.method == 'strong':
            f_out = self.linear(f_in) * self.linear_embedding(cond_vec)
        elif self.method == 'pure':
            f_out = self.bilinear(f_in, cond_vec)
        
        if add_vec is not None:
            f_out[add_index] += add_vec

        return f_out

class LinearHead(nn.Module):
    def __init__(self, layer_dims, nonlinearity='relu', normalization='batch', flatten=False):
        super(LinearHead, self).__init__()
        
        # Activation function
        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        elif nonlinearity == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        
        # Normalization layers
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d
        elif normalization == 'layer':
            self.norm = nn.LayerNorm
        elif normalization == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")
        
        self.flatten = flatten

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            
            if i < len(layer_dims) - 2:  # Skip normalization and activation for the last layer
                if self.norm is not None:
                    layers.append(self.norm(layer_dims[i+1]))
                layers.append(self.activation)
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, backbone_output, data=None):
        embedding = backbone_output

        if self.flatten:
            embedding = embedding.flatten(start_dim=1)

        return self.mlp(embedding)


class GraphLevelPredictionHead(nn.Module):
    def __init__(self, layer_dims, nonlinearity='relu', normalization='batch', pool_type='mean', dropout=0.0):
        super(GraphLevelPredictionHead, self).__init__()

        self.pool_type = pool_type
        self.pred = LinearHead(layer_dims, nonlinearity, normalization)
        self.dropout = dropout

        if pool_type not in ['mean', 'sum', 'peptide_sum']:
            raise ValueError(f"Unsupported pool type: {pool_type}")

    def forward(self, backbone_output, data=None):
        embedding = backbone_output

        if self.pool_type == "mean":
            out = global_mean_pool(embedding, data.batch)
        elif self.pool_type == "sum":
            out = global_add_pool(embedding, data.batch)
        elif self.pool_type == "peptide_sum":
            p_embedding = embedding[data.entity == 1]
            if self.dropout:
                p_embedding = F.dropout(p_embedding, p=self.dropout, training=self.training)
            out = self.pred(p_embedding)
            return global_add_pool(out, data.batch[data.entity == 1])

        return self.pred(out)

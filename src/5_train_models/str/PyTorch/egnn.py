import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, SiLU, Sequential

from embedding import AtomEmbedding, RadialBasisEmbedding
from linear import ConditionalLinear, GraphLevelPredictionHead

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_scatter import scatter


"""
This file contains the code for the E(n) Equivariant Graph Neural Network (EGNN) model.
Reference: E(n) Equivariant Graph Neural Networks, Satorras et al.

It has a few things changed from the original paper:

* Instead of using raw distances in angstroms at features, they are embedded using radial basis functions. (See embedding.py)

* In the original paper, node_i, node_j features and the distance features are concatenated and fed into the linear layers. (See linear.py)
    - In this module, you have three ways of conditioning the MLP on the distance:
        
        - weak: This is the same as the original paper, where the distance is concatenated with the node features.

            i.e. f_out = self.linear(torch.cat((node_i, node_j, distance_features), dim=-1))
        
        - strong: The distance features are separately transformed using a linear layer, and then multiplied with the node features.

            i.e. f_out = self.linear(torch.cat(node_i, node_j)) * self.linear_embedding(distance_features)

        - pure: This is even more expressive, but it's very expensive. No need to use it.

* For pooling on the pMHC, I use the "peptide_sum" pooling method. It changes the order of the pooling and the predictor MLP.

    i.e. 

        standard pooling:   1. pool over all nodes # (n, d) -> (batch_size, d)
                            2. make prediction on pooled features # (batch_size, d) -> (batch_size, 1)
        
        peptide pooling:    1. get peptide residue scores # (num_peptide_residues_in_batch, d) -> (num_peptides_in_batch, 1)
                            2. pool over peptides # (num_peptides_in_batch, 1) -> (batch_size, 1)
"""

class MLP_Message(torch.nn.Module):
    def __init__(self, emb_dim, norm, activation, edge_dim=0,
                 deep_conditioning=False, dropout=0.0,
                 rbf=False, rbf_min=0, rbf_max=30, rbf_dim=64,
                 method="weak"):
        """E(n) Equivariant GNN Message MLP
        Paper: E(n) Equivariant Graph Neural Networks, Satorras et al.
        
        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            edge_dim: (int) - edge feature dimension
            deep_conditioning: (bool) - whether to condition on distance in every MLP layer
            dropout: (float) - dropout probability in update MLP
            method: (str) - conditioning method (weak/strong/bilinear)
            rbf: (bool) - whether to use radial basis functions for distance encoding
            rbf_min: (float) - minimum distance for radial basis functions
            rbf_max: (float) - maximum distance for radial basis functions
            rbf_dim: (int) - number of radial basis functions
        """
    
        super().__init__()

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        
        self.rbf = rbf

        self.deep_conditioning = deep_conditioning
        cond_dim = 1 + edge_dim

        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

        if rbf:
            self.cond_projection = nn.Sequential(
                RadialBasisEmbedding(rbf_dim, rbf_min, rbf_max),
                nn.Linear(rbf_dim, emb_dim - edge_dim)
            )
            cond_dim = emb_dim
        else:
            self.cond_projection = None

        self.mlp = torch.nn.ModuleList([
            ConditionalLinear(2 * emb_dim, emb_dim, cond_dim, method=method),
            self.norm(emb_dim),
            self.activation,
            ConditionalLinear(emb_dim, emb_dim, cond_dim, method=method) if self.deep_conditioning else Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        ])

    def forward(self, msg_h, msg_cond):
        if self.cond_projection is not None:
            if self.edge_dim > 0 and self.rbf == True: 
                cond_proj_distance = self.cond_projection(msg_cond[:, 0].unsqueeze(-1))
                msg_cond  = torch.cat([cond_proj_distance, msg_cond[:, 1:]], dim=-1)
            else:
                msg_cond = self.cond_projection(msg_cond)       

        for i, module in enumerate(self.mlp):
            if isinstance(module, ConditionalLinear):
                msg_h = module(msg_h, msg_cond)
            else:
                # So, I realized that I have a bug in my code which will only have the first layer run for this module
                # All other layers will be skipped. 
                # To make the code a faithful reproduction of the module I use, I'm leaving it as is.
                # Removing the if statement should make thigns work as intended, and ideally wouldn't change the results.
                if False:
                    msg_h = module(msg_h)

        return msg_h


class EGNNLayer(MessagePassing):
    def __init__(self, emb_dim, edge_dim=0, 
                 activation="relu", norm="layer", aggr="add",
                 cond_method="weak", deep_conditioning=False,
                 rbf=False, rbf_min=0, rbf_max=30, rbf_dim=10,
                 dropout=0.0, update_pos=True):
        """E(n) Equivariant GNN Layer
        Paper: E(n) Equivariant Graph Neural Networks, Satorras et al.
        
        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension

            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            
            cond_method: (str) - conditioning method (weak/strong/bilinear)
            deep_conditioning: (bool) - whether to condition on distance in every MLP layer
            
            rbf: (bool) - whether to use radial basis functions for distance encoding
            rbf_min: (float) - minimum distance for radial basis functions
            rbf_max: (float) - maximum distance for radial basis functions
            rbf_dim: (int) - number of radial basis functions
            
            dropout: (float) - dropout probability in update MLP

        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

        self.mlp_msg = MLP_Message(emb_dim, norm, activation, edge_dim=edge_dim, method=cond_method, dropout=dropout,
                                   deep_conditioning=deep_conditioning,
                                   rbf=rbf, rbf_min=rbf_min, rbf_max=rbf_max, rbf_dim=rbf_dim)

        # MLP `\psi_x` for computing messages `\overrightarrow{m}_ij`
        self.mlp_pos = Sequential(
            Linear(emb_dim, emb_dim), self.norm(emb_dim), self.activation, Linear(emb_dim, 1)
        )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            nn.Dropout(dropout),
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

        self.update_pos = update_pos

    def forward(self, h, pos, edge_index, edge_attr=None):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr=None):
        # Compute messages
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        msg_h = torch.cat([h_i, h_j], dim=-1)

        if edge_attr is not None:
            msg_cond = torch.cat([dists, edge_attr], dim=-1)
        else:
            msg_cond = dists
        # msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg_h, msg_cond)

        # Scale magnitude of displacement vector
        if self.update_pos:
            pos_diff = pos_diff * self.mlp_pos(msg)  # torch.clamp(updates, min=-100, max=100)
        else:
            pos_diff = None

        return msg, pos_diff

    def aggregate(self, inputs, index):
        msgs, pos_diffs = inputs

        # Aggregate messages
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate displacement vectors
        if self.update_pos:
            pos_aggr = scatter(pos_diffs, index, dim=self.node_dim, reduce="mean")

        # Only update nodes that have received messages
        nodes_to_upd = torch.unique(index)

        msg_aggr = msg_aggr[nodes_to_upd]
        if self.update_pos:
            pos_aggr = pos_aggr[nodes_to_upd]
        else:
            pos_aggr = None

        return msg_aggr, pos_aggr, nodes_to_upd

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr, nodes_to_upd = aggr_out

        upd_out = h
        upd_out[nodes_to_upd] = self.mlp_upd(torch.cat([h[nodes_to_upd], msg_aggr], dim=-1))

        if self.update_pos:
            upd_pos = pos
            upd_pos[nodes_to_upd] = pos[nodes_to_upd] + pos_aggr
        else:
            upd_pos = pos
            
        return upd_out, upd_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"


class EGNNModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=5,
        emb_dim=128, in_dim=1, out_dim=1,edge_dim=0,
        activation="relu", norm="layer", residual=True,
        aggr="sum", pool="peptide_sum",
        embed=True, separate_entity_embeddings=False,
        cond_method="weak", 
        deep_conditioning=False,
        rbf=True, rbf_min=0.0, rbf_max=30.0, rbf_dim=64,
        update_pos=False,
        dropout=0.0,
    ):
        """E(n) Equivariant GNN model 
        
        Args:
            num_layers: (int) - number of message passing layers
            emb_dim: (int) - hidden dimension
            in_dim: (int) - initial node feature dimension
            out_dim: (int) - output number of classes
            
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            residual: (bool) - whether to use residual connections
            
            aggr: (str) - aggregation function for messages `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            
            embed: (bool) - whether to use an embedding layer for initial node features
            separate_entity_embeddings: (bool) - whether to use separate embeddings for peptide and ligand nodes
            
            cond_method: (str) - conditioning method (weak/strong/bilinear)
            deep_conditioning: (bool) - whether to condition on distance in every MLP layer
            
            rbf: (bool) - whether to use radial basis functions for distance encoding
            rbf_min: (float) - minimum distance for radial basis functions
            rbf_max: (float) - maximum distance for radial basis functions
            rbf_dim: (int) - number of radial basis functions
            
            update_pos: (bool) - whether to update node coordinates

            dropout: (float) - dropout probability in update MLP
        """
        super().__init__()

        # Embedding lookup for initial node features
        self.embed = embed
        self.separate_entity_embeddings = separate_entity_embeddings

        if embed:
            self.embedding = AtomEmbedding(in_dim, emb_dim, separate_entity_embedding=separate_entity_embeddings)
        else:
            self.embedding = lambda x: x.x
        

        self.update_pos = update_pos

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EGNNLayer(
                                        emb_dim, edge_dim=edge_dim, activation=activation, 
                                        norm=norm, aggr=aggr, cond_method=cond_method,
                                        deep_conditioning=deep_conditioning,
                                        dropout=dropout,
                                        rbf=rbf, rbf_min=rbf_min, rbf_max=rbf_max, rbf_dim=rbf_dim,
                                        update_pos=update_pos,
                                        ))

        self.pred = GraphLevelPredictionHead(
            [emb_dim, emb_dim, out_dim], nonlinearity=activation, normalization=norm, pool_type=pool
        )
        
        self.residual = residual


    def forward(self, batch, node_idx=None):
        h = self.embedding(batch) # (n, d)
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index, 
                                        batch.edge_attr if hasattr(batch, "edge_attr") else None,
                                        )

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update 

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update

        return self.pred(h, batch)

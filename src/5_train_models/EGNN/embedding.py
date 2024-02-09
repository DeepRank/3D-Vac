import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

class AtomEmbedding(nn.Module):
    """
    This is your standard residue/atom type embedding.

    Optionally you can make separate embeddings for the ligand and protein.

    """

    def __init__(self, num_atoms, embedding_dim=128, separate_entity_embedding=False):
        super().__init__()

        self.num_atoms = num_atoms
        self.embedding_dim = embedding_dim

        if separate_entity_embedding:
            self.embedding = nn.Embedding(num_atoms * 2, embedding_dim)
        else:    
            self.embedding = nn.Embedding(num_atoms, embedding_dim)

        self.separate_entity_embedding = separate_entity_embedding

    def forward(self, data):    
        if self.separate_entity_embedding:
            x = data.x * 2 + data.entity
        else:
            x = data.x

        # data.x = self.embedding(x)
        return self.embedding(x)  

class RadialBasisEmbedding(nn.Module):
    """
    This module is used to embed the distance between two atoms into a feature vector.

    Check out SchNet or DimeNet for more information.
    """


    def __init__(self, num_radial, radial_min, radial_max):
        super(RadialBasisEmbedding, self).__init__()

        self.num_radial = num_radial
        self.radial_min = radial_min
        self.radial_max = radial_max

        centers = torch.linspace(radial_min, radial_max, num_radial).view(-1, 1)
        widths = (centers[1] - centers[0]) * torch.ones_like(centers)

        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, distance):
        rbf = torch.exp(-((distance.squeeze() - self.centers) ** 2) / (2 * self.widths ** 2))
        return rbf.transpose(-1, -2)
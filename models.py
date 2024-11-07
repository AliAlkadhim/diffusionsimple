import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


from sklearn.datasets import make_moons
# Set numpy random seed for reproducibility
np.random.seed(42)
import time
from jetnet.datasets import JetNet
from jetnet.utils import jet_features
from configs import *

# Epsilon neural network model
class Epsilon(nn.Module):
    """Neural network model for noise prediction
    epsilon_theta: x_t, t -> noise
    """

    def __init__(
        self,
        nfeatures,
        ntargets,
        nlayers,
        hidden_size,
        activation,
        time_embedding_dim=128,
    ):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim

        # Time embedding layer
        #project time embedding dimension to hidden size
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_size),
            self._get_activation(activation),
        )

        # Input layer for x_t
        self.input_layer = nn.Sequential(
            nn.Linear(nfeatures, hidden_size),
            self._get_activation(activation),
        )

        # Hidden layers
        layers = []
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self._get_activation(activation))
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, ntargets)

    def _get_activation(self, activation):
        activations = {
            "LeakyReLU": nn.LeakyReLU(negative_slope=0.3),
            "ReLU": nn.ReLU(),
            "PReLU": nn.PReLU(),
            "ReLU6": nn.ReLU6(),
            "ELU": nn.ELU(),
            "SELU": nn.SELU(),
            "CELU": nn.CELU(),
        }
        return activations.get(activation, nn.ReLU())  # Default to ReLU if activation is not found

    def forward(self, x, t_embedding):
        # Combine x and t embeddings
        x = self.input_layer(x) 
        t_emb = self.time_mlp(t_embedding)
        h = x + t_emb  # Alternatively, you can concatenate and use another layer 
        h = self.hidden_layers(h)
        return self.output_layer(h)
    
class GNN(nn.Module):
    def __init__(
        self,
        nfeatures,
        ntargets,
        nlayers,
        hidden_size,
        activation,
        time_embedding_dim=128,
    ):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim

        # Time embedding layer
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_size),
            self._get_activation(activation),
        )

        # Input layer for x_t
        self.conv1 = GCNConv(nfeatures, hidden_size)
        
        # Hidden layers
        self.convs = nn.ModuleList()
        #we're using ModuleList because an ordinary list wont be optimized by PyTorch
        for _ in range(nlayers - 1):
            self.convs.append(GCNConv(hidden_size, hidden_size))
        
        self.activation = self._get_activation(activation)
        # Output layer
        self.output_layer = nn.Linear(hidden_size, ntargets)

    def _get_activation(self, activation):
            activations = {
                "LeakyReLU": nn.LeakyReLU(negative_slope=0.3),
                "ReLU": nn.ReLU(),
                "PReLU": nn.PReLU(),
                "ReLU6": nn.ReLU6(),
                "ELU": nn.ELU(),
                "SELU": nn.SELU(),
                "CELU": nn.CELU(),
            }
            return activations.get(activation, nn.ReLU())  # Default to ReLU if activation is not found

    def forward(self, x, edge_index, t_embedding, batch):
        """
        edge_index is a tensor containing pairs of connected node indices
        
        batch is a tensor containing the batch indices for each node
        """
        # Process time embedding
        t_emb = self.time_mlp(t_embedding)
        #t_emb now has shape [num_nodes, hidden_size] 
        # Initial convolution
        h = self.conv1(x, edge_index)
        #Considers the graph structure defined by edge_index
        h = self.activation(h)
        
        h = h + t_emb  # Add time embedding
        
        # Hidden layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.activation(h)

        h = global_mean_pool(h, batch)
        
        return self.output_layer(h)
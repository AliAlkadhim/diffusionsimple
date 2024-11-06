import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import torch.optim as optim
import torch.nn as nn
from sklearn.datasets import make_moons
# Set numpy random seed for reproducibility
np.random.seed(42)
import time
import jetnet
from jetnet.datasets import JetNet
from jetnet.utils import jet_features

from jetnet_diffusion import *
from configs import *


    # Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


particle_data = np.load('datasets/jetnet/particle_data.npy')
print(f'particle_data.shape: {particle_data.shape}')

jet_data = np.load('datasets/jetnet/jet_data.npy')

print('using substructure')
jet_data = jet_data[:SUBSET]
particle_data = particle_data[:SUBSET, :, :]
print(f'jet_data.shape: {jet_data.shape}')
print(f'particle_data.shape: {particle_data.shape}')
num_types = len(data_args["jet_type"])
print(f'num_types: {num_types}')
type_indices = {jet_type: JetNet.JET_TYPES.index(jet_type) for jet_type in data_args["jet_type"]}# these are special indices for gluon, top quark, and W boson jets
print(f'type_indices: {type_indices}') 


x0 = particle_data
x0_red = x0[:SUBSET]

flattened_x0 = x0_red.reshape(-1, 3)
flat_x0_red = torch.tensor(flattened_x0, dtype=torch.float32).to(device)
print(f'x0.shape={flat_x0_red.shape}')
# Total time steps T
T = 1000
# Noise scheduler beta_t, which goes from beta_1 to beta_T
beta_1 = 1e-4
beta_T = 0.02
beta = torch.linspace(beta_1, beta_T, T).to(device)
alpha_ = 1 - beta
alpha_bar = torch.cumprod(alpha_, dim=0)

    # Train the model
epsilon_theta, x0_mean, x0_std = train_substructure(epochs=epochs, 
                x0=flat_x0_red, 
                alpha_bar=alpha_bar, 
                T=T, 
                device=device)
#!/usr/bin/env python
# coding: utf-8

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


print(f"Particle features: {JetNet.ALL_PARTICLE_FEATURES}")
print(f"Jet features: {JetNet.ALL_JET_FEATURES}")



particle_data, jet_data = JetNet.getData(**data_args)

print(f'jet_data.shape: {jet_data.shape}')
print(f'particle_data.shape: {particle_data.shape}')
print(f"\nJet features of first jet\n{data_args['jet_features']}\n{jet_data[0]}")


np.save('datasets/jetnet/particle_data.npy', particle_data)
print('saving datasets/jetnet/particle_data.npy')   
np.save('datasets/jetnet/jet_data.npy', jet_data)
print('saving datasets/jetnet/jet_data.npy')

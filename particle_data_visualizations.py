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


selected_observables = ['mass', 'pt', 'eta']
selected_observables_labels = ['$m^{rel}$', '$p_T^{rel}$', '$\eta^{rel}$']
jet_types = ['g', 't', 'w']


def plot_particle_data_observables(jet_data, particle_data, type_indices, selected_observables, jet_types):
    fig, ax = plt.subplots(1,len(selected_observables), figsize=(10,10))
    for ind, observable in enumerate(selected_observables):
        for jet_type in jet_types:
            substruc_observable = get_substructure_feature(jet_data=jet_data,      
                particle_data=particle_data, 
                type_indices=type_indices,
                selected_observable=observable,
                jet_type=jet_type)
            ax[ind].hist(substruc_observable, histtype='step', label=jet_type)
            ax[ind].set_xlabel(selected_observables_labels[ind])
            ax[ind].legend()

    plt.tight_layout()  
    plt.show()

plot_particle_data_observables(jet_data=jet_data, 
particle_data=particle_data, type_indices=type_indices, selected_observables=selected_observables, jet_types=jet_types)
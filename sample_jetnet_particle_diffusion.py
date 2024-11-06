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



x0_mean = flat_x0_red.mean(dim=0).cpu().numpy()
x0_std = flat_x0_red.std(dim=0).cpu().numpy()

# load model

epsilon_theta = Epsilon(
nfeatures=flat_x0_red.shape[1],
ntargets=flat_x0_red.shape[1],
nlayers=n_layers,
hidden_size=hidden_size,
activation='ReLU',
time_embedding_dim=time_embedding_dim,
).to(device)

epsilon_theta.load_state_dict(torch.load(f'models/weights/particles_epsilon_theta_{epochs}_epochs_MLP.pth'))

start_time = time.time()
x_sample_1 = sample_one(
    model=epsilon_theta,
    x0_shape=flat_x0_red.shape,
    alpha=alpha_,
    beta=beta,
    alpha_bar=alpha_bar,
    T=T_sample_1,
    device=device,
)
x_sample_1 = x_sample_1.cpu().numpy()
end_time = time.time()
print(f'Time to sample one feature of shape={x_sample_1.shape}: {end_time - start_time:.2f} seconds')

    ### denormalize and plot

x_sample_1_denormalized = x0_mean + x_sample_1 * x0_std

x_sample_1_denormalized = x_sample_1_denormalized.reshape(x0_red.shape)

fig, ax = plt.subplots(1,len(selected_observables), figsize=(10,10))
for ind, observable in enumerate(selected_observables):
    for jet_type in jet_types:
        substruc_observable_generated = get_substructure_feature(jet_data=jet_data,      
            particle_data=x_sample_1_denormalized, 
            type_indices=type_indices,
            selected_observable=observable,
            jet_type=jet_type)
        ax[ind].hist(substruc_observable_generated, histtype='step', label=jet_type + ' generated')

        substruc_observable_real = get_substructure_feature(jet_data=jet_data,      
        particle_data=x0_red, 
        type_indices=type_indices,
        selected_observable=observable,
        jet_type=jet_type)
        ax[ind].hist(substruc_observable_real, histtype='step', label=jet_type + ' real')
        
        ax[ind].set_xlabel(selected_observables_labels[ind])
        ax[ind].legend()
plt.tight_layout()  
plt.show()
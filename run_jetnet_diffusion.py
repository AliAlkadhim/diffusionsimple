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


    # Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


print(f"Particle features: {JetNet.ALL_PARTICLE_FEATURES}")
print(f"Jet features: {JetNet.ALL_JET_FEATURES}")

data_args = {
"jet_type": ["g", "t", "w"],  # gluon, top quark, and W boson jets
"data_dir": "datasets/jetnet",
# only selecting the kinematic features
"particle_features": ["etarel", "phirel", "ptrel"],
"num_particles": 30,
# "jet_features": ["type", "pt", "eta", "mass"],
"jet_features": ["type","pt", "eta", "mass"],
    # "download": True,
    "download": True,
}

particle_data, jet_data = JetNet.getData(**data_args)

print(f'jet_data.shape: {jet_data.shape}')
print(f'particle_data.shape: {particle_data.shape}')
print(f"\nJet features of first jet\n{data_args['jet_features']}\n{jet_data[0]}")

    # decide if you want to analyze the jet data or the particle data
jets_or_particles = "particles"

if jets_or_particles == 'jets':
    x0 = jet_data
else:
    x0 = particle_data

SUBSET = int(1e4)
# SUBSET = -1
x0_red = x0[:SUBSET]
    

#################### JETS ####################
if jets_or_particles == 'jets':
    fig, ax = plt.subplots(1,3, figsize=(10,10))
    for i in range(3):
        ax[i].hist(x0_red[:,i],bins=100);

    plt.show()
    
    ############################################################
    
    x0_red = torch.tensor(x0_red, dtype=torch.float32).to(device)
    print(f'x0.shape={x0_red.shape}')
    # Total time steps T
    T = 1000
    # Noise scheduler beta_t, which goes from beta_1 to beta_T
    beta_1 = 1e-4
    beta_T = 0.02
    beta = torch.linspace(beta_1, beta_T, T).to(device)
    alpha_ = 1 - beta
    alpha_bar = torch.cumprod(alpha_, dim=0)
    
    plot_forward_xt_jet(T_ex=T, x0=x0_red.cpu(), alpha_bar=alpha_bar.cpu(), N_plots=5)
    
    # Train the model
    epochs = 50
    epsilon_theta, x0_mean, x0_std = train(epochs=epochs, 
                    x0=x0_red, 
                    alpha_bar=alpha_bar, 
                    T=T, 
                    device=device)
    
    # SAMPLE
    T_sample_1 = 1000
    # calculate time to sample one feature
    start_time = time.time()
    x_sample_1 = sample_one(
        model=epsilon_theta,
        x0_shape=x0_red.shape,
        alpha=alpha_,
        beta=beta,
        alpha_bar=alpha_bar,
        T=T_sample_1,
        device=device,
    )
    x_sample_1 = x_sample_1.numpy()
    end_time = time.time()
    print(f'Time to sample one feature of shape={x_sample_1.shape}: {end_time - start_time:.2f} seconds')
    
    ### denormalize and plot
    fig, ax = plt.subplots(1,3, figsize=(10,10))

    x_sample_1_denormalized = x0_mean + x_sample_1 * x0_std
    for i in range(3):
        ax[i].hist(x_sample_1_denormalized[:,i],bins=100, alpha = 0.4, label = 'generated samples', histtype='step')
        ax[i].hist(x0_red.numpy()[:,i],                     bins=100, alpha = 0.4, label = 'original samples', histtype='step')
        ax[i].legend(fontsize=15)
    plt.show()
        
        
#################### PARTICLES ####################
elif jets_or_particles == "particles":
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
    
    ############################################################
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
    epochs = 50
    epsilon_theta, x0_mean, x0_std = train_substructure(epochs=epochs, 
                    x0=flat_x0_red, 
                    alpha_bar=alpha_bar, 
                    T=T, 
                    device=device)
    
     # SAMPLE
    T_sample_1 = 1000
    # calculate time to sample one feature
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
    
    # plot samples
    x_sample_1_denormalized = x_sample_1_denormalized.reshape(x0_red.shape)
    
    fig, ax = plt.subplots(1,len(selected_observables), figsize=(10,10))
    for ind, observable in enumerate(selected_observables):
        for jet_type in jet_types:
            substruc_observable = get_substructure_feature(jet_data=jet_data,      
                particle_data=x_sample_1_denormalized, 
                type_indices=type_indices,
                selected_observable=observable,
                jet_type=jet_type)
            ax[ind].hist(substruc_observable, histtype='step', label=jet_type)
            ax[ind].set_xlabel(selected_observables_labels[ind])
            ax[ind].legend()
    plt.tight_layout()  
    plt.show()
    
    ## evaluation metrics
    # fpnd_score = jetnet.evaluation.fpnd(x_sample_1_denormalized)
    # print(f'fpnd_score: {fpnd_score}')
    
    
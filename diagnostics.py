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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
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


sample_filename = f'samples/particles_sample_1_T_sample_{T_sample_1}_epochs_{epochs}_subset_{str(SUBSET)}.npy'
x_sample_1_denormalized = np.load(sample_filename)

print(f'x_sample_1_denormalized.shape: {x_sample_1_denormalized.shape}')


jet_type = 'g'
type_selector = jet_data[:, 0] == type_indices[jet_type]  # type_indices: {'g': 0, 't': 2, 'w': 3}
particle_data_g = particle_data[type_selector]
fpnd_score_jetnet = jetnet.evaluation.fpnd(particle_data_g, jet_type="g")
print(f'fpnd_score for {jet_type} jet type between jetnet and jetnet: {fpnd_score_jetnet}')

# see https://jetnet.readthedocs.io/en/latest/pages/metrics.html
for jet_type in jet_types:
    # jet_type = 'g'
    type_selector = jet_data[:, 0] == type_indices[jet_type]  # type_indices: {'g': 0, 't': 2, 'w': 3}
    particles_jet_type = particle_data[type_selector]

    generated_jet_type_particles = x_sample_1_denormalized[type_selector]

    fpnd_score = jetnet.evaluation.fpnd(generated_jet_type_particles, jet_type="g")
    print(f'fpnd_score for {jet_type} jet type: {fpnd_score}')

    w1p_score = jetnet.evaluation.w1p(particles_jet_type,generated_jet_type_particles) 
    print(f'w1p score for {jet_type} jet type: {w1p_score}')

    w1m_score = jetnet.evaluation.w1m(particles_jet_type,generated_jet_type_particles) 
    print(f'w1m score for {jet_type} jet type: {w1m_score}')

    w1efp_score = jetnet.evaluation.w1efp(particles_jet_type,generated_jet_type_particles, use_particle_masses=True) 
    print(f'w1efp score for {jet_type} jet type: {w1efp_score}')

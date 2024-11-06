import torch
import numpy as np
# SUBSET = int(1e5)
SUBSET = None

# Set device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

selected_observables = ['mass', 'pt', 'eta']
selected_observables_labels = ['$m^{rel}$', '$p_T^{rel}$', '$\eta^{rel}$']
jet_types = ['g', 't', 'w']


n_layers = 4
hidden_size = 128
time_embedding_dim = 128

epochs = 500

# Total time steps T
T = 1000
beta_1 = 1e-4
beta_T = 0.02
beta = torch.linspace(beta_1, beta_T, T).to(device)
alpha_ = 1 - beta
alpha_bar = torch.cumprod(alpha_, dim=0)

T_sample_1 = 60
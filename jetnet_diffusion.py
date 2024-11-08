#!/usr/bin/env python
# coding: utf-8

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
from models import *

def time_type_of_func(_func=None):
    def timer(func):
        """Print the runtime of the decorated function"""
        import functools
        import time

        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time

            print(f"this arbirary function took {run_time:.4f} secs")
            return value

        return wrapper_timer

    if _func is None:
        return timer
    else:
        return timer(_func)
    

#sinusoidal embedding function for time steps
def sinusoidal_embedding(timesteps, embedding_dim):
    """
    Creates sinusoidal embeddings for the time steps.
    Look at "Attention is all you need" for more details.
    """
    device = timesteps.device
    half_dim = embedding_dim // 2
    # emb is of shape [half_dim]
    emb = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(half_dim, device=device) / (half_dim - 1)
    )
    # if timesteps had original shape [batch_size], timesteps[:, None] will make it [batch_size, 1]
    emb = timesteps[:, None] * emb[None, :]
    # emb is now of shape [1, batch_size, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb



# Function to get x_t and noise epsilon at time t
# @time_type_of_func()
def get_x_t(x0, t, alpha_bar):
    """
    Get the noisy data x_t and the noise that was added to it at time t
    x_t ~ q(x_t | x_{t-1})
    """
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(x0)
        
    epsilon = torch.randn_like(x0)
    
    # Ensure t is a tensor of shape [batch_size]
    alpha_bar_t = alpha_bar[t].unsqueeze(1)  # Shape: [batch_size, 1]
    
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    
    xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * epsilon
    return xt, epsilon

# Function to plot the forward diffusion process
@time_type_of_func()
def plot_forward_xt_jet(T_ex, x0, alpha_bar, N_plots=10):
    fig, axs = plt.subplots(1, N_plots, figsize=(20, 10))
    den = T_ex // N_plots
    for idx, t in enumerate(range(0, T_ex, den)):
        print(f't={t}')
        t_batch = torch.full((x0.shape[0],), t, dtype=torch.long)
        xt, _ = get_x_t(x0, t_batch, alpha_bar)
        axs[idx].set_title(f't = {t}')
        axs[idx].hist(xt[:,2].numpy().flatten(), bins=100)
    plt.tight_layout()
    plt.show()
    
    
    
def plot_forward_xt(T_ex, x0, alpha_bar, N_plots=10):
    fig, axs = plt.subplots(1, N_plots, figsize=(20, 2))
    den = T_ex // N_plots
    for idx, t in enumerate(range(0, T_ex, den)):
        print(f't={t}')
        t_batch = torch.full((x0.shape[0],), t, dtype=torch.long)
        xt, _ = get_x_t(x0, t_batch, alpha_bar)
        axs[idx].set_title(f't = {t}')
        axs[idx].scatter(xt[:, 0], xt[:, 1], s=10, alpha=0.4)
    plt.tight_layout()
    plt.show()

# Function to sample new data points using the trained model
# @time_type_of_func()
def sample_one(model, x0_shape, alpha, beta, alpha_bar, T, device, model_type):
    model.eval()
    with torch.no_grad():
        
        x_t = torch.randn(x0_shape).to(device)
        
        for i in reversed(range(T)):
            print(f't: {i}')
            t = torch.full((x0_shape[0],), i, dtype=torch.long).to(device)
            
            t_embedding = sinusoidal_embedding(t, model.time_embedding_dim).to(device)
            if model_type=='mlp':
                predicted_noise = model(x=x_t, t_embedding=t_embedding)
            elif model_type=='gnn':
                num_nodes = x_t.shape[0]
                edge_index = torch.combinations(torch.arange(num_nodes)).t().to(device)
                batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
                predicted_noise = model(x_t, edge_index=edge_index, t_embedding=t_embedding,
                                        batch=batch)
            
            alpha_t = alpha[t].unsqueeze(1).to(device)
            
            alpha_bar_t = alpha_bar[t].unsqueeze(1).to(device)
            
            beta_t = beta[t].unsqueeze(1).to(device)
            
            sigma_t = torch.sqrt(beta_t)

            if i > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)

            x_t = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            ) + sigma_t * z
            
    model.train()
    return x_t


def get_sample_filename(T_sample_1, epochs, n_layers, hidden_size, model_type, subset):
    sample_filename = f'samples/particles_sample_1_T_sample_{T_sample_1}_epochs_{epochs}_nlayers_{n_layers}_hidden_size_{hidden_size}_subset_{str(SUBSET)}_type_{model_type}.npy'
    return sample_filename


def get_model_filename(epochs, n_layers, hidden_size, model_type):
    model_filename = f'models/weights/particles_epsilon_theta_{epochs}_epochs_MLP_nlayers_{n_layers}_hidden_size_{hidden_size}.pth'
    return model_filename

# Training function
@time_type_of_func()
def train(model_type,epochs, x0, alpha_bar, T, device):
    """
    Train the model to predict the noise
    we pass x0 = flattened_x0 to train()
    """
    if model_type == 'mlp':
        model = Epsilon(
            nfeatures=x0.shape[1],
        ntargets=x0.shape[1],
        nlayers=4,
        hidden_size=128,
        activation='ReLU',
        time_embedding_dim=128,
        ).to(device)
    elif model_type == 'gnn':
        model = GNN(
            nfeatures=x0.shape[1],
            ntargets=x0.shape[1],
            nlayers=4,
            hidden_size=128,
            activation='ReLU',
            time_embedding_dim=128,
        ).to(device)

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()

    batch_size = 256  
    x0_mean = x0.mean(dim=0)
    x0_std = x0.std(dim=0)
    normalized_x0 = (x0 - x0_mean) / x0_std
    dataset = torch.utils.data.TensorDataset(normalized_x0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x0, in dataloader:
            batch_x0 = batch_x0.to(device)
            # batch_x0 
            t = torch.randint(low=0, high=T, size=(batch_x0.shape[0],), device=device)
            
            optimizer.zero_grad()
            
            xt, noise = get_x_t(batch_x0, t, alpha_bar)
            
            t_embedding = sinusoidal_embedding(t, model.time_embedding_dim).to(device)
            
            if model_type == 'mlp':
                predicted_noise = model(x=xt, t_embedding=t_embedding)
            elif model_type == 'gnn':
                num_nodes = xt.shape[0]
                
                edge_index = torch.combinations(torch.arange(num_nodes)).t().to(device)

                predicted_noise = model(x=xt, edge_index=edge_index, t_embedding=t_embedding)
            
            loss = loss_fn(predicted_noise, noise)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x0.size(0)
       
        avg_loss = epoch_loss / len(dataloader.dataset)
       
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}')

    return model, x0_mean.cpu().numpy(), x0_std.cpu().numpy() 



@time_type_of_func()
def train_substructure(model_type,epochs, x0, alpha_bar, T, device):
    """
    Train the model to predict the noise
    """
    if model_type == 'mlp':
        model = Epsilon(
            nfeatures=x0.shape[1],
        ntargets=x0.shape[1],
        nlayers=n_layers,
        hidden_size=hidden_size,
        activation='ReLU',
        time_embedding_dim=time_embedding_dim,
        ).to(device)
    elif model_type == 'gnn':
        model = GNN(
            nfeatures=x0.shape[1],
            ntargets=x0.shape[1],
            nlayers=n_layers,
            hidden_size=hidden_size,
            activation='ReLU',
            time_embedding_dim=time_embedding_dim,
        ).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()

    batch_size = BATCHSIZE
    x0_mean = x0.mean(dim=0)
    x0_std = x0.std(dim=0)
    normalized_x0 = (x0 - x0_mean) / x0_std
    dataset = torch.utils.data.TensorDataset(normalized_x0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
    # shuffle=True
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x0, in dataloader:
            batch_x0 = batch_x0.to(device)
            
            t = torch.randint(low=0, high=T, size=(batch_x0.shape[0],), device=device)
            
            optimizer.zero_grad()
            
            xt, noise = get_x_t(batch_x0, t, alpha_bar)
            
            t_embedding = sinusoidal_embedding(t, model.time_embedding_dim).to(device)
            # t_embedding will be of shape [batch_size, time_embedding_dim]
            if model_type == 'mlp':
                predicted_noise = model(x=xt, t_embedding=t_embedding)
            
            elif model_type == 'gnn':
                num_nodes = xt.shape[0]
                
                edge_index = torch.combinations(torch.arange(num_nodes)).t().to(device)
                batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
                # all nodes are connected to eachother within each batch, so assign each node to its corresponding batch index
                predicted_noise = model(x=xt, edge_index=edge_index, t_embedding=t_embedding,
                                        batch=batch)
            
            loss = loss_fn(predicted_noise, noise)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x0.size(0)
       
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f'epoch: {epoch}, loss: {avg_loss:.6f}')

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}')

    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('models/weights'):
        os.makedirs('models/weights')
    model_filename = get_model_filename(epochs, n_layers, hidden_size, model_type)

    torch.save(model.state_dict(), model_filename)
    return model, x0_mean.cpu().numpy(), x0_std.cpu().numpy() 


def get_substructure_feature(jet_data,      
            particle_data, 
            type_indices,
            selected_observable,
            jet_type):
    type_selector = jet_data[:, 0] == type_indices[jet_type]  # type_indices: {'g': 0, 't': 2, 'w': 3}
    
    if selected_observable not in ['mass', 'pt', 'eta']:
        raise ValueError(f"Invalid observable: {selected_observable}. Must be one of: mass, pt, eta")
        
    if jet_type not in type_indices:
        raise ValueError(f"Invalid jet type: {jet_type}. Must be one of: {list(type_indices.keys())}")
        
    try:
        selected_particles = particle_data[type_selector]
        selected_feature = jet_features(selected_particles)[selected_observable]
        return selected_feature
    except Exception as e:
        print(f"Error calculating {selected_observable} for {jet_type} jets:")
        print(f"Particle data shape: {particle_data.shape}")
        print(f"Selected particles shape: {selected_particles.shape}")
        raise e






# Main execution block
if __name__ == '__main__':
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
    "jet_features": ["pt", "eta", "mass"],
        # "download": True,
        "download": False,
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

    SUBSET = int(1e5)
    # SUBSET = -1
    x0_red = x0[:SUBSET]
    
    #################### JETS ####################
    if jets_or_particles == 'jets':
        fig, ax = plt.subplots(1,3, figsize=(10,10))
        for i in range(3):
            ax[i].hist(x0_red[:,i],bins=100);

        plt.show()
        
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
            
            
    #################### PARTICLES ####################
    elif jets_or_particles == "particles":
        num_types = len(data_args["jet_type"])
        type_indices = {jet_type: JetNet.JET_TYPES.index(jet_type) for jet_type in data_args["jet_type"]}# these are special indices for gluon, top quark, and W boson jets
        relmass_g = get_substructure_feature(jet_data=jet_data,      
            particle_data=particle_data, 
            type_indices=type_indices,
            selected_observable='mass',
            jet_type='g')
        print(relmass_g)
        # fig, ax = plt.subplots(1,3, figsize=(10,10))

        # plt.hist(relmass_g, bins=np.linspace(0, 0.2, 100), histtype="step", label='g')
        
        # plt.show()
        
        # for i, jet_type in enumerate(data_args["jet_type"]):
            # plot_particle_data_fist(jet_data=jet_data,      
            # particle_data=particle_data, 
            # ax=ax[i], 
            # type_indices=type_indices,
            # selected_observable='mass')
        
              
        
        # x0_red = torch.tensor(x0_red, dtype=torch.float32).to(device)
        # print(f'x0.shape={x0_red.shape}')
        # # Total time steps T
        # T = 1000
        # # Noise scheduler beta_t, which goes from beta_1 to beta_T
        # beta_1 = 1e-4
        # beta_T = 0.02
        # beta = torch.linspace(beta_1, beta_T, T).to(device)
        # alpha_ = 1 - beta
        # alpha_bar = torch.cumprod(alpha_, dim=0)
        
        # plot_forward_xt_jet(T_ex=T, x0=x0_red.cpu(), alpha_bar=alpha_bar.cpu(), N_plots=5)
        
        # # Train the model
        # epochs = 50
        # epsilon_theta, x0_mean, x0_std = train(epochs=epochs, 
        #               x0=x0_red, 
        #               alpha_bar=alpha_bar, 
        #               T=T, 
        #               device=device)
        
        # # SAMPLE
        # T_sample_1 = 1000
        # # calculate time to sample one feature
        # start_time = time.time()
        # x_sample_1 = sample_one(
        #     model=epsilon_theta,
        #     x0_shape=x0_red.shape,
        #     alpha=alpha_,
        #     beta=beta,
        #     alpha_bar=alpha_bar,
        #     T=T_sample_1,
        #     device=device,
        # )
        # x_sample_1 = x_sample_1.numpy()
        # end_time = time.time()
        # print(f'Time to sample one feature of shape={x_sample_1.shape}: {end_time - start_time:.2f} seconds')
        
        # ### denormalize and plot
        # fig, ax = plt.subplots(1,3, figsize=(10,10))

        # x_sample_1_denormalized = x0_mean + x_sample_1 * x0_std
        # for i in range(3):
        #     ax[i].hist(x_sample_1_denormalized[:,i],bins=100, alpha = 0.4, label = 'generated samples', histtype='step')
        #     ax[i].hist(x0_red.numpy()[:,i],                     bins=100, alpha = 0.4, label = 'original samples', histtype='step')
        #     ax[i].legend(fontsize=15)
            
    
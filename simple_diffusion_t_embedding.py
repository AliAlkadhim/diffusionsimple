#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from sklearn.datasets import make_moons
# Set numpy random seed for reproducibility
np.random.seed(42)
import time

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
    emb = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(half_dim, device=device) / (half_dim - 1)
    )
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

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
        axs[idx].hist(xt[:,2].numpy().flatten())
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
def sample_one(model, x0_shape, alpha, beta, alpha_bar, T, device):
    model.eval()
    with torch.no_grad():
        
        x_t = torch.randn(x0_shape).to(device)
        
        for i in reversed(range(T)):
            
            t = torch.full((x0_shape[0],), i, dtype=torch.long).to(device)
            
            t_embedding = sinusoidal_embedding(t, model.time_embedding_dim).to(device)
            
            predicted_noise = model(x_t, t_embedding)
            
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

# Training function
@time_type_of_func()
def train(epochs, x0, alpha_bar, T, device):
    """
    Train the model to predict the noise
    """
    model = Epsilon(
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
    dataset = torch.utils.data.TensorDataset(x0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x0, in dataloader:
            batch_x0 = batch_x0.to(device)
            
            t = torch.randint(low=0, high=T, size=(batch_x0.shape[0],), device=device)
            
            optimizer.zero_grad()
            
            xt, noise = get_x_t(batch_x0, t, alpha_bar)
            
            t_embedding = sinusoidal_embedding(t, model.time_embedding_dim).to(device)
            
            predicted_noise = model(xt, t_embedding)
            
            loss = loss_fn(predicted_noise, noise)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x0.size(0)
       
        avg_loss = epoch_loss / len(dataloader.dataset)
       
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}')

    return model

# Main execution block
if __name__ == '__main__':
    # Set device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Generate the dataset
    X, _ = make_moons(n_samples=3000, 
                      noise=0, 
                      random_state=0)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.title('Original Data')
    plt.show()

    x0 = torch.tensor(X, dtype=torch.float32).to(device)
    print(f'x0.shape={x0.shape}')

    # Total time steps T
    T = 1000
    # Noise scheduler beta_t, which goes from beta_1 to beta_T
    beta_1 = 1e-4
    beta_T = 0.02
    beta = torch.linspace(beta_1, beta_T, T).to(device)
    alpha_ = 1 - beta
    alpha_bar = torch.cumprod(alpha_, dim=0)

    # Plot the forward diffusion process
    plot_forward_xt(T_ex=T, x0=x0.cpu(), alpha_bar=alpha_bar.cpu(), N_plots=10)

    # Train the model
    epochs = 100  
    epsilon_theta = train(epochs=epochs, 
                          x0=x0, 
                          alpha_bar=alpha_bar, 
                          T=T, 
                          device=device)

    # Generate new samples
    X_original = x0.numpy()
    
    T_sample_1 = 1000
    # calculate time to sample one feature
    start_time = time.time()
    x_sample_1 = sample_one(
        model=epsilon_theta,
        x0_shape=x0.shape,
        alpha=alpha_,
        beta=beta,
        alpha_bar=alpha_bar,
        T=T_sample_1,
        device=device,
    )
    x_sample_1 = x_sample_1.numpy()
    end_time = time.time()
    print(f'Time to sample one feature of shape={x_sample_1.shape}: {end_time - start_time:.2f} seconds')
    

    T_sample_2 = 50
    x_sample_2 = sample_one(
        model=epsilon_theta,
        x0_shape=x0.shape,
        alpha=alpha_,
        beta=beta,
        alpha_bar=alpha_bar,
        T=T_sample_2,
        device=device,
    )
    x_sample_2 = x_sample_2.numpy()
    
    # Plot the generated samples and original data
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].scatter(x_sample_1[:, 0], x_sample_1[:, 1], s=10, label='Generated Samples', alpha=0.4)
    axs[0].scatter(X_original[:, 0], X_original[:, 1], s=10, label='Original Data', alpha=0.4)
    axs[0].legend()
    axs[0].set_title(r'Generated Samples vs Original Data, $T_{sample}=T_{train}=$ %d' % T_sample_1)

    axs[1].scatter(x_sample_2[:, 0], x_sample_2[:, 1], s=10, label='Generated Samples', alpha=0.4)
    axs[1].scatter(X_original[:, 0], X_original[:, 1], s=10, alpha=0.4, label='Original Data')
    axs[1].legend()
    axs[1].set_title(r'Generated Samples vs Original Data, $T_{sample}=$ %d' % T_sample_2)

    plt.show()

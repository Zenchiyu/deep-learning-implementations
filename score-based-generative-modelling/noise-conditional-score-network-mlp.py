import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from toy_distributions_generative_modelling import ToyGen
from tqdm import tqdm


class NoiseConditionalScoreNetwork(nn.Module):
    """
    This network is trained to predict the (Stein)
    score functions of perturbed data distributions
    for different noise levels.

    The score of a distribution p(x) is grad_x log p(x)
    Therefore, the output of the network is of same shape
    as the input (if we ignore the noise parameter)

    Just an MLP with softplus activation functions.

    No layer norm, no U-Net.
    """
    def __init__(self, input_dim, hidden_dim=128) -> None:
        super().__init__()
        # A Multi-Layer Perceptron
        self.layers = nn.Sequential(
            nn.Linear(input_dim+1, hidden_dim),
            nn.Softplus(),  # to follow a bit the paper
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim) 
            # I don't use a softplus for the last layer
            # since gradients can be negative!
        )
    
    def forward(self, x, std):
        x_new = torch.cat([x, std.expand(x.shape[0], 1)], dim=1)
        return self.layers(x_new)

def train_ncsn(model, real_data,
               stds,
               nb_epochs=100,
               lr=1e-4, 
               batch_size=256):

    train_loader = DataLoader(real_data, batch_size=batch_size,
                              num_workers=4, pin_memory=True,
                              shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop, training the score function
    for e in tqdm(range(nb_epochs)):
        for X in train_loader:
            # Call the score network for different noise levels
            loss = 0
            for std in stds:
                # Perturbe the input to
                # - solve manifold hypothesis issue
                # - poor predictions for low density regions
                X_noisy = X + torch.normal(0, std, X.shape)
                
                # Predict scores of noisy-perturbed
                y_hat = model(X_noisy, std)

                # Denoising score matching objective
                l = 0.5*(y_hat+(X_noisy-X)/std**2).pow(2).mean()

                # Combine into the unified objective
                w = std**2
                loss += w*l
            # Unified objective
            loss /= len(stds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def sample(n_samples, model, stds, input_dim, eps, T,
           hypercube_len=2, track_paths=False):
    """
    Annealed Langevin dynamics using trained score network

    eps: related to step size in Langevin dynamics
    T: number of steps before changing to next noise level
    """
    # Sample from prior distribution (here uniform)
    x = hypercube_len*torch.rand(size=(n_samples, input_dim))-hypercube_len/2
    x_paths = torch.empty((T*len(stds)+1, *x.shape))
    x_paths[0] = x

    i = 1
    for std in stds:  # should be in decreasing order
        alpha = eps*std**2/stds[-1]**2  # step size
        for _ in range(T):
            z = torch.randn(size=(n_samples, input_dim))
            x += alpha/2*model(x, std) + torch.sqrt(alpha)*z
            x_paths[i] = x
            i += 1
    if track_paths:
        return x, x_paths

    return x

def plot_estimated_score_fct(model, std, xmin, xmax, ymin, ymax):
    # Plot estimated vector field for particular std
    X = torch.linspace(xmin, xmax, 20)
    Y = torch.linspace(ymin, ymax, 20)
    XX, YY = torch.meshgrid(X, Y)
    data = torch.stack([XX.flatten(), YY.flatten()], dim=1)
    scores = model(data, std)
    U = scores[:, 0].reshape(XX.shape)
    V = scores[:, 1].reshape(YY.shape)
    q = plt.quiver(XX, YY, U, V, alpha=0.3)
    return XX, YY, U, V

if __name__ == "__main__":
    N_train = 10_000
    nb_epochs = 1000
    lr = 1e-4
    batch_size = 128

    # Noise levels & Langevin meta-params
    eps = 1e-4
    std_1, std_L, L = 5, 0.01, 10
    T = 100

    
    M = torch.tensor([[1, 0.5], [0.5, 1]])
    f_inv = lambda x: x + torch.stack([torch.sign(x[:, 0]), torch.zeros(x.shape[0])], dim=1)
    # f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M)
    
    # f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M) + torch.stack([torch.sign((x @ M)[:, 0]), torch.zeros(x.shape[0])], dim=1)
    # f_inv = lambda x: x @ torch.tensor([[1, 0.5], [0.5, 1]])
    
    real_data = ToyGen(N_train, f_inv)
    
    # Noise levels following a geometric sequence
    # std_1 = std_L*common_ratio**(L-1)
    # therefore, common_ratio = (std_1/std_L)**(1/(L-1))
    common_ratio = (std_1/std_L)**(1/(L-1))
    stds = std_L*common_ratio**torch.arange(L-1, -1, -1)
    # Decreasing order, from std_1 to std_L
    
    # Instantiate our model
    model = NoiseConditionalScoreNetwork(input_dim=2)

    # Train our model
    train_ncsn(model, real_data, stds,
               nb_epochs=nb_epochs, lr=lr, batch_size=batch_size)

    x, x_paths = sample(1_000, model, stds, input_dim=2, eps=eps, T=T,
                        hypercube_len=3,
                        track_paths=True)

    with torch.no_grad():
        real_data.plot()
        real_data.plot_transformed_circles()
        plt.plot(x[:, 0], x[:, 1], '.r')
        _ = plot_estimated_score_fct(model, torch.tensor(std_L), -4, 4, -4, 4)
        plt.plot(x_paths[250:, 0, 0], x_paths[250:, 0, 1], 'k', linewidth=0.5)
        plt.plot(x_paths[-1, 0, 0], x_paths[-1, 0, 1], 'gx', markersize=7, markeredgewidth=3)
        
        plt.show()
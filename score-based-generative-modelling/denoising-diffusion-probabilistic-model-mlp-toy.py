import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from toy_distributions_generative_modelling import ToyGen


class NoisePredictor(nn.Module):
    # An implementation of eps_theta(x, t)
    def __init__(self, input_dim, hidden_dim, T):
        super().__init__()
        self.T = T
        self.layers = nn.Sequential(
            nn.Linear(input_dim+1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        t = expand_dim(t.expand(x.shape[0], *t.shape[1:]), x.ndim)/self.T
        xt = torch.cat([x, t], dim=1)
        return self.layers(xt)

def expand_dim(x, dim):
    # Append dimensions to x
    return x.view(x.shape+(1, )*(dim-x.ndim))

def train_ddpm(train_loader, model, criterion, optimizer, **config):
    nb_epochs = config["nb_epochs"]
    beta_1 = config["beta_1"]
    beta_T = config["beta_T"]
    T = config["T"]

    # Linear variance schedule
    betas = torch.linspace(beta_1, beta_T, T)
    alpha_bars = torch.cumprod(1-betas, dim=0)

    batch_seen = 0  # step
    for e in tqdm(range(nb_epochs)):
        acc_loss = 0
        for X in train_loader:
            N = X.shape[0]

            # Sample times
            time_cond = torch.randint(T, size=(N, ))
            a_bars = alpha_bars[time_cond].view(N, 1)
            
            # Noise and noisy input
            noise = torch.randn(X.shape)
            X_noisy = torch.sqrt(a_bars)*X+torch.sqrt(1-a_bars)*noise  # from re-param trick
            
            output = model(X_noisy, time_cond)
            loss = criterion(output, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_loss += loss.item()
            batch_seen += 1

def sample(model, nb_samples, config, track_path=False):
    """
    Reverse diffusion process
    """
    beta_1 = config["beta_1"]
    beta_T = config["beta_T"]
    T = config["T"]

    # Linear variance schedule
    betas = torch.linspace(beta_1, beta_T, T)
    alphas = 1-betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    # Fixed reverse process stds (isotropic gaussian)
    stds = torch.sqrt(betas)

    x = torch.randn(size=(nb_samples, 2))
    x_paths = torch.empty((T+1, *x.shape))
    x_paths[0] = x

    for i, t in enumerate(torch.arange(T-1, -1, -1)):
        b, a, a_bar, s = betas[t], alphas[t], alpha_bars[t], stds[t]
        z = 0 if t == 0 else torch.randn(size=x.shape)
        mean = (x-b/torch.sqrt(1-a_bar)*model(x, t))/torch.sqrt(a)
        x = mean + s*z
        x_paths[i+1] = x
    
    if track_path:
        return x, x_paths
    return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "device": device,
        "lr": 1e-4,
        "nb_epochs": 1000,  # 1000
        "batch_size": 128,
        "dataset": "toy",
        # Variance schedule for DDPM
        "beta_1": 1e-4,  # 1e-5
        "beta_T": 0.02,
        # Number of discrete time steps
        "T": 1000,
        "N_train": 10_000,
    }
    config["dataset_name"] = "toy"

    # run = wandb.init(
    #     project="ddpm",
    #     # track hyperparameters and run metadata
    #     config=config
    # )

    # Dataloaders
    
    M = torch.tensor([[1, 0.5], [0.5, 1]])
    f_inv = lambda x: x + torch.stack([torch.sign(x[:, 0]), torch.zeros(x.shape[0])], dim=1)
    # f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M)
    
    # f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M) + torch.stack([torch.sign((x @ M)[:, 0]), torch.zeros(x.shape[0])], dim=1)
    # f_inv = lambda x: x @ torch.tensor([[1, 0.5], [0.5, 1]])
    
    real_data = ToyGen(config["N_train"], f_inv)
    train_loader = DataLoader(real_data, batch_size=config["batch_size"],
                              shuffle=True, num_workers=4,
                              pin_memory=torch.cuda.is_available())

    # Model and criterion
    model = NoisePredictor(input_dim=2, hidden_dim=128, T=config["T"])
    criterion = nn.MSELoss()
    model.to(device=device)
    criterion.to(device=device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_ddpm(train_loader, model, criterion, optimizer, **config)

    x, x_paths = sample(model, 1000, config, track_path=True)
            
    with torch.no_grad():
        real_data.plot()
        real_data.plot_transformed_circles()
        plt.plot(x[:, 0], x[:, 1], '.r')
        plt.plot(x_paths[250:, 0, 0], x_paths[250:, 0, 1], 'k', linewidth=0.5)
        plt.plot(x_paths[-1, 0, 0], x_paths[-1, 0, 1], 'gx', markersize=7, markeredgewidth=3)
        
        plt.show()

    # wandb.finish()

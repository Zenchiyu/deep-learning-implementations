import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from test_case_generative_modelling import TestCaseGen, plot_transformed_circles
from tqdm import tqdm


if __name__ == "__main__":
    nb_epochs = 500  # 500
    dim = 2  # input dim
    z_dim = 16  # latent dimension, not the same as input dim unlike normalizing flows
    
    eta = 1e-3
    batch_size = 100
    hidden_dim = 100

    N = 10_000
    
    # Test case
    M = torch.tensor([[1, 0.5], [0.5, 1]])
    # f_inv = lambda x: x + torch.stack([torch.sign(x[:, 0]), torch.zeros(x.shape[0])], dim=1)
    f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M)
    
    # f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M) + torch.stack([torch.sign((x @ M)[:, 0]), torch.zeros(x.shape[0])], dim=1)
    # f_inv = lambda x: x @ torch.tensor([[1, 0.5], [0.5, 1]])
    real_data = TestCaseGen(N, f_inv=f_inv)
    train_loader = DataLoader(real_data, shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=4, pin_memory=True)
    
    # Generator, discriminator using Multi-Layer Perceptrons
    G = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
    D = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    
    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=eta)
    opt_D = torch.optim.Adam(D.parameters(), lr=eta)

    # Training loop, alternating optimization of generator G and discriminator D
    # for each different batch
    avg_D_of_real = []
    avg_D_of_fake = []
    for e in tqdm(range(nb_epochs)):
        avg_D_of_real_epoch = 0
        avg_D_of_fake_epoch = 0
        for b, X in enumerate(train_loader):
            n = X.shape[0]
            # Create n fake samples using the generator
            Z = torch.normal(0, 1, size=(n, z_dim))
            fake_X = G(Z)

            # Pass it through Discriminator
            probs_real_given_fake = D(fake_X)
            probs_fake_given_fake = 1-probs_real_given_fake
            probs_real_given_real = D(X)

            if b % 2 == 0:
                # Update generator using -log(D(G(z))) trick
                # instead of log(1-D(G(z)))
                loss = -torch.log(probs_real_given_fake).mean()
                # Intuitively, we want that the discriminator gives
                # high proba of our fake samples being real!

                # From the course
                # loss = (1-probs_real_given_fake).log().mean()
                opt_G.zero_grad()
                loss.backward()
                opt_G.step()
            else:
                # Update discriminator, min binary cross entropy
                loss = -0.5*(torch.log(probs_real_given_real).mean()+torch.log(probs_fake_given_fake).mean())
                
                # From the course
                # loss = -(1-probs_real_given_fake).log().mean()\
                #        -probs_real_given_real.log().mean()
                opt_D.zero_grad()
                loss.backward()
                opt_D.step()

            avg_D_of_real_epoch = (1-n/N)*avg_D_of_real_epoch + n/N*probs_real_given_real.mean()
            avg_D_of_fake_epoch = (1-n/N)*avg_D_of_fake_epoch + n/N*probs_real_given_fake.mean()
        avg_D_of_real.append(avg_D_of_real_epoch.item())
        avg_D_of_fake.append(avg_D_of_fake_epoch.item())

    with torch.no_grad():
        z = torch.randn(size=(1000, z_dim))
        x_hat = G(z)
        
        real_data.plot()
        real_data.plot_transformed_circles()
        plt.plot(x_hat[:, 0], x_hat[:, 1], '.r')
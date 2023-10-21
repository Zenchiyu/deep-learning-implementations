import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from toy_distributions_generative_modelling import ToyGen, plot_transformed_circles
from tqdm import tqdm


class NVPCouplingLayer(nn.Module):
    def __init__(self, b: torch.Tensor,
                 map_s: callable,
                 map_t: callable) -> None:
        super().__init__()
        # self.b = b  
        # the course uses register_buffer because of model persistence
        self.register_buffer('b', b.unsqueeze(0))
        self.map_s = map_s
        self.map_t = map_t

    def forward(self, x: torch.Tensor, ldj: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # ldj stands for log determinant jacobian
        bx = self.b*x
        s, t = self.map_s(bx), self.map_t(bx)

        y = bx + (1-self.b)*(x*torch.exp(s) + t)
        new_ldj = ldj + ((1-self.b)*s).sum(dim=1)
        return y, new_ldj
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        bx = self.b*y
        s, t = self.map_s(bx), self.map_t(bx)
        return bx + (1-self.b)*(y-t)*torch.exp(-s)
    
class NVPNet(nn.Module):
    def __init__(self, dim, hidden_dim, depth) -> None:
        """
        - dim: dimension of the data
        - hidden_dim: dim of the hidden layer in the
        MLP for s and for t.
        - depth: number of coupling layers
        """
        super().__init__()
        self.layers = nn.ModuleList()
        b = torch.zeros(dim)

        for d in range(depth):
            # Create b, s, t
            if d % 2 == 0:
                b[torch.randint(0, dim, size=(dim//2,))] = 1
            else:
                b = 1 - b  # odd layers, flip bits in b

            # Multi-Layer Perceptrons (why not ReLU?)
            s = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, dim)
            )
            t = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, dim)
            )
                
            # Append a NVP coupling layer
            self.layers.append(NVPCouplingLayer(b.clone(), s, t))
        
    def forward(self, x, ldj):
        for layer in self.layers:
            x, ldj = layer(x, ldj)
        return x, ldj

    def inverse(self, y: torch.Tensor):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

if __name__ == "__main__":
    nb_epochs = 1000 # 500
    eta = 1e-2
    batch_size = 100

    M = torch.tensor([[1, 0.5], [0.5, 1]])
    # f_inv = lambda x: x + torch.stack([torch.sign(x[:, 0]), torch.zeros(x.shape[0])], dim=1)
    f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M)
    # f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M) + torch.stack([torch.sign((x @ M)[:, 0]), torch.zeros(x.shape[0])], dim=1)
    # f_inv = lambda x: x @ torch.tensor([[1, 0.5], [0.5, 1]])
    real_data = ToyGen(10_000, f_inv=f_inv)
    train_loader = DataLoader(real_data, shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=4, pin_memory=True)
    dim = 2
    model = NVPNet(dim=dim, hidden_dim=2, depth=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    for e in tqdm(range(nb_epochs)):
        for X in train_loader:
            n = X.shape[0]
            X_out, ldj = model(X, 0)

            log_gaussian = -0.5*((X_out**2).sum(dim=1)+dim*np.log(np.pi))
            loss = -(log_gaussian + ldj).sum()  # minimize negative log likelihood

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        z = torch.randn(size=(1000, 2))
        x_hat = model.inverse(z)
        
        real_data.plot()
        real_data.plot_transformed_circles()
        plt.plot(x_hat[:, 0], x_hat[:, 1], '.r')
        plot_transformed_circles(model.inverse, color="r")

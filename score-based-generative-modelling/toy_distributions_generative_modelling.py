import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class ToyGen(Dataset):
    def __init__(self, nb: int, f_inv: callable) -> None:
        super().__init__()
        # 2D Gaussian
        Z = torch.normal(0, 1, size=(nb, 2))
        self.nb = nb
        self.f_inv = f_inv
        self.X = self.f_inv(Z)

    def __len__(self) -> int:
        return self.nb
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        return self.X[index]
    
    def plot(self, ax=None):
        if ax is None:
            plt.plot(self.X[:, 0], self.X[:, 1], '.', markersize=1)
        else:
            ax.plot(self.X[:, 0], self.X[:, 1], '.', markersize=1)

    def plot_transformed_circles(self, ax=None, color="b"):
        plot_transformed_circles(self.f_inv, ax=ax, color=color)

def plot_transformed_circles(f_inv, ax=None, color="b"):
    n_radius = 10
    angles = torch.linspace(0, 2*np.pi, 1001)[:-1].expand(n_radius, 1000).flatten()
    radius = torch.linspace(0.2, 3, n_radius).expand(1000, n_radius).T.flatten()
    circles = radius[:, None]*torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    transformed_circles = f_inv(circles)
    if ax is None:
        plt.plot(transformed_circles[:, 0], transformed_circles[:, 1],
                 '-', color=color)
    else:
        ax.plot(transformed_circles[:, 0], transformed_circles[:, 1],
                '-', color=color)

if __name__ == "__main__":
    # f_inv = lambda x: torch.log(torch.tanh(x))
    # f_inv = lambda x: torch.tanh(x)
    # f_inv = lambda x: torch.sigmoid(x)
    f_inv = lambda x: x @ torch.tensor([[1, 0.5], [0.5, 1]])
    # M = torch.tensor([[1, 0.5], [0.5, 1]])
    # f_inv = lambda x: torch.stack([torch.ones(x.shape[0]), torch.sign((x @ M)[:, 0])], dim=1)*(x @ M)
    data = ToyGen(10000, f_inv=f_inv)
    data.plot()
    # Plotting transformed circles
    data.plot_transformed_circles()
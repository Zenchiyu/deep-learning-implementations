from typing import Any
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose, Lambda

from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), embed_dim=16):
        super().__init__()
        C, H, W = input_shape
        self.conv1 = nn.Conv2d(C, 32, kernel_size=5)  # H -> H-5+1=H-4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # H-4 -> H-4-5+1=H-8
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)  # H-8 -> (H-8-4)//2+1=(H-12)//2+1
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)  # (H-12)//2+1 -> ((H-12)//2+1-3)//2+1=((H-12)//2 - 2)//2+1
        # Up to here we have: N x 64 x ((H-12)//2 - 2)//2+1
        self.conv5 = nn.Conv2d(64, 2*embed_dim, kernel_size=(((H-12)//2 - 2)//2+1,
                                                           ((W-12)//2 - 2)//2+1))
        # we get N x 2*embed_dim x 1 x 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

class Decoder(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), embed_dim=16):
        super().__init__()
        C, H, W = input_shape
        self.tconv5 = nn.ConvTranspose2d(32, C, kernel_size=5)  # H -> H-5+1=H-4
        self.tconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5)  # H-4 -> H-4-5+1=H-8
        self.tconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2)  # H-8 -> (H-8-4)//2+1=(H-12)//2+1
        self.tconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)  # (H-12)//2+1 -> ((H-12)//2+1-3)//2+1=((H-12)//2 - 2)//2+1
        # Up to here we have: N x 64 x ((H-12)//2 - 2)//2+1
        self.tconv1 = nn.ConvTranspose2d(embed_dim, 64, kernel_size=(((H-12)//2 - 2)//2+1,
                                                           ((W-12)//2 - 2)//2+1))
        # we go from N x 2*embed_dim x 1 x 1 to N x *input_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No pooling
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = F.relu(self.tconv5(x))
        return x

class VAE(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), embed_dim=16):
        super().__init__()
        self.encode = Encoder(input_shape, embed_dim)
        self.decode = Decoder(input_shape, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu_f = self.encode(x)[:, :x.shape[1]//2]
        return self.decode(mu_f)

def get_datasets(dataset: torch.utils.data.Dataset,
                 normalize_features: bool=True) -> tuple[Dataset, Dataset]:
    dataset_name = dataset.__name__.lower()

    train_data = dataset(f"../data/{dataset_name}/train",
                         train=True, download=True)
    test_data = dataset(f"../data/{dataset_name}/test",
                         train=False, download=True)
    
    # Mean and std
    if type(train_data.data) == torch.Tensor:
        train_x = train_data.data.to(dtype=torch.float)/255
    else:
        # assume numpy array
        train_x = torch.from_numpy(train_data.data).to(dtype=torch.float)/255
    
    if len(train_data.data.shape) == 4:
        N, H, W, C = train_data.data.shape
        train_x = train_x.permute(0, 3, 1, 2)
    elif len(train_data.data.shape) == 3:
        N, H, W = train_data.data.shape
        C = 1
        train_x = train_x.view(N, C, H, W)
    
    if normalize_features:
        mean = train_x.mean(dim=0)  # in one go
        std = train_x.std(dim=0)
    else:
        mean = train_x.mean()
        std = train_x.std()

    target_transform = Lambda(lambda y: torch.tensor(y))
    transform = Compose([ToTensor(),
                         Lambda(lambda x: x.view(C, H, W)),
                         Lambda(lambda x: x.sub(mean).div(std))
    ])
    train_data.transform = transform
    test_data.transform = transform
    train_data.target_transform = target_transform
    test_data.target_transform = target_transform

    return train_data, test_data


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_epochs = 5 # 300  # 600
    batch_size = 256
    embed_dim = 16 # Don't use 32.. that's too big
    beta = 1  # beta=1 for VAE, otherwise just beta-VAE
    eta = 1e-4  # learning rate

    dataset = CIFAR10  # MNIST
    dataset_name = dataset.__name__

    train_data, test_data = get_datasets(dataset=dataset, normalize_features=False)
    train_loader = DataLoader(train_data, num_workers=4, pin_memory=True,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=4, pin_memory=True,
                             batch_size=batch_size)

    N, (C, H, W) = len(train_loader), train_loader.dataset[0][0].shape

    # Model(s)
    model = VAE(input_shape=(C, H, W), embed_dim=embed_dim)
    model.to(device=device)
    # Number of parameters
    # sum(map(lambda x: x.numel(), model.parameters()))
    # or equivalently
    # sum([torch.prod(torch.tensor(p.size())) for p in model.parameters()])

    # Criterion
    # criterion =   # nn.MSELoss() used for reconstruction loss
    # criterion.to(device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)


    # Training loop
    acc_losses = torch.zeros(n_epochs)
    for e in tqdm(range(n_epochs)):
        for (X, _) in train_loader:  # ignore labels
            X = X.to(device=device)
            ## Encoder predicts parameters of a gaussian
            mu_f, logvar_f = model.encode(X).split(embed_dim, dim=1)
            # KL between q(Z|X=X) and p(Z)=N(0, Id) to force get a gaussian
            kl_loss = -0.5*(1+logvar_f-mu_f**2-torch.exp(logvar_f))  # KL between two gaussians
            kl_loss = kl_loss.sum()/X.shape[0]

            # Generating a latent vector using the reparam trick
            std_f = torch.exp(0.5*logvar_f)
            Z = torch.empty_like(mu_f).normal_(0, 1)*std_f + mu_f

            ## Decoder predicts a conditional expectation
            X_hat = model.decode(Z)
            # MSE loss, reconstruction loss
            fit_loss = torch.pow(X_hat - X, 2)/2
            fit_loss = fit_loss.sum()/X.shape[0]
            # fit_loss = torch.mean((X_hat-X).pow(2))/2

            loss = beta*kl_loss + fit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc_losses[e] += loss.item()

        print(f"Epoch {e} acc_loss: {acc_losses[e]}")

    # Generate some stuffs !
    with torch.no_grad():
        nrows, ncols = 5, 12

        z = torch.normal(0, 1, size=(ncols*nrows, embed_dim, 1, 1))
        x_hat = model.decode(z)
        
        torchvision.utils.save_image(x_hat, f"dlc-7-4-variational-auto-encoders-sampling-{dataset_name}.png",
                                     nrow=ncols, pad_value=1.0)
    
        # Save model
        general_checkpoint = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "n_epochs": n_epochs,
                              "batch_size": batch_size,
                              "embed_dim": embed_dim,
                              "beta": beta,
                              "eta": eta,
                              "dataset": dataset,
                              "acc_losses": acc_losses}
        
        torch.save(general_checkpoint, f"general_checkpoint_beta_vae_{dataset_name}.pth")
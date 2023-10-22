import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import wandb

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Lambda, Compose, ToTensor 
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, input_dim, output_shape) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape
        # N x input_dim x 1 x 1 -> increase the spatial dimensions using interpolations and conv
        # self.m = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),  # 2x2
        #     nn.Conv2d(input_dim, 256, kernel_size=3, padding=1, bias=False),  # same padding conv
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),  # 4x4
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),  # 8x8
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),  # 16x16
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),  # 32x32
        #     nn.Conv2d(32, output_shape[0], kernel_size=3, padding=1),
        #     nn.Tanh()
        # )
        self.m = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 2x2
            nn.Conv2d(input_dim, 256, kernel_size=3, padding=1, bias=False),  # same padding conv
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # 8x8
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # 32x32
            nn.Conv2d(128, output_shape[0], kernel_size=3, padding=1),
            nn.Tanh()
        )
        # Since BN has a "bias" parameters, we set the bias in the conv to False
        # Note that the last "layer" doesn't have a BN layer
        # Interpolations used to increase spatial dim:
        # nn.Upsample is like F.interpolate
        # Convolutions used to change number of channels
        
    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.m = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, bias=False),  # (32-4)/2+1 = 15
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=False),  # (15-5)/2+1=6
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=False),  # (6-4)/2+1=2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=2, stride=1, bias=False),  # (2-2)+1=1
            nn.Sigmoid()
        )
        # At this point we have a N x 1 x 1 x 1 tensor 
        # Since BN has a "bias" parameters, we set the bias in the conv to False
        # Note that the first "layer" doesn't have a BN layer

    def forward(self, x):
        return self.m(x).squeeze(-1).squeeze(-1)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    nb_epochs = 50  # 100 # 50 # 5
    eta = 1e-4
    betas = (0.5, 0.999)  # beta1 coming from DCGAN paper
    batch_size = 256
    latent_dim = 1000 # 16

    run = wandb.init(
        # set the wandb project where this run will be logged
        project="standard-gan-cnn", #"dcgan-like-cnn",
        # track hyperparameters and run metadata
        config={
            "learning_rate": eta,
            "betas": betas,
            "epochs": nb_epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim
        }
    )

    target_transform = Lambda(lambda y: torch.tensor(y))
    train_data = CIFAR10("../data/cifar-10/train", train=True,
                         download=True, target_transform=target_transform)
    test_data = CIFAR10("../data/cifar-10/test", train=False,
                         download=True, target_transform=target_transform)

    # Normalization
    train_x = (torch.from_numpy(train_data.data)/255).permute(0, 3, 1, 2)
    
    N, C, H, W = train_x.shape
    input_shape = (3, H, W)

    # mean = train_x.mean(dim=0)
    # std = train_x.std(dim=0)
    # transform = Compose([ToTensor(), Lambda(lambda x: x.sub(mean).div(std))])
    
    # Scaling between -1 and 1
    min = train_x.min(dim=0).values
    max = train_x.max(dim=0).values
    transform = Compose([ToTensor(), Lambda(lambda x: 2*(x.sub(min).div(max-min))-1)])

    train_data.transform = transform
    test_data.transform = transform

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                              num_workers=4, pin_memory=True)
    
    # Models: generator, discriminator
    model_G = Generator(input_dim=latent_dim, output_shape=input_shape)
    model_D = Discriminator(input_shape)
    model_G.to(device=device)
    model_D.to(device=device)

    # Criterions
    criterion = nn.BCELoss()
    criterion.to(device=device)

    # Optimizers
    optimizer_G = optim.Adam(model_G.parameters(), lr=eta, betas=betas)
    optimizer_D = optim.Adam(model_D.parameters(), lr=eta, betas=betas)

    # Wandb Log
    run.watch(model_G, model_D, log="all", log_graph=True)

    # Training loop
    model_G.train()
    model_D.train()
    loss_G = 0
    loss_D = 0

    for e in tqdm(range(nb_epochs)):
        for b, (X_real, _) in enumerate(train_loader):
            # Move data on device
            X_real = X_real.to(device=device)

            current_batch_size = X_real.shape[0]

            # Generate fake examples
            Z = X_real.new(size=(current_batch_size, latent_dim)).normal_(0, 1)
            X_fake = model_G(Z)

            D_on_real = model_D(X_real)
            D_on_fake = model_D(X_fake)

            output_D = torch.cat([D_on_real, D_on_fake], dim=0)
            labels = torch.ones(output_D.shape)
            labels[current_batch_size:] = 0

            if b % 3 < 2:  # Train generator twice more
                # Train the generator using the -log(D(G(Z))) trick
                # by flipping labels
                loss = criterion(D_on_fake, torch.ones((current_batch_size, 1)))
                optimizer_G.zero_grad()
                loss.backward()
                optimizer_G.step()
                loss_G = loss.item()
            else:
                # Train the discriminator
                loss = criterion(output_D, labels)
                optimizer_D.zero_grad()
                loss.backward()
                optimizer_D.step()
                loss_D = loss.item()

        

        with torch.no_grad():
            model_G.eval()
            model_D.eval()
            # Changes the behavior of Batch Norm layers

            ncols = 12
            nrows = 5
            z = torch.randn(size=(ncols*nrows, latent_dim))
            x_fake = model_G(z)
            # plt.imshow(x_fake[0].permute(1, 2, 0))
            torchvision.utils.save_image(x_fake/2+0.5, "dlc-11-1-gan-sampling.png",
                                        nrow=ncols, pad_value=1.0)
            
            for b, (X_real_test, _) in enumerate(test_loader):
                D_on_real_test = model_D(X_real_test)
                break

            wandb.log({"epoch": e,
                       "sampling": wandb.Image("dlc-11-1-gan-sampling.png"),
                       "D_on_real_avg_last_batch": D_on_real.mean(),
                       "D_on_real_std_last_batch": D_on_real.std(),
                       "D_on_fake_avg_last_batch": D_on_fake.mean(),
                       "D_on_fake_std_last_batch": D_on_fake.std(),
                       "D_on_real_test_avg_first_batch": D_on_real_test.mean(),
                       "D_on_real_test_std_first_batch": D_on_real_test.std(),
                       "loss_D_last_batch": loss_D,
                       "loss_G_last_batch": loss_G})
            
            general_checkpoint = {"model_G_state_dict": model_G.state_dict(),
                        "optimizer_G_state_dict": optimizer_G.state_dict(),
                        "model_D_state_dict": model_D.state_dict(),
                        "optimizer_D_state_dict": optimizer_D.state_dict(),
                        "nb_epochs": nb_epochs,
                        "epoch": e,
                        "batch_size": batch_size,
                        "latent_dim": latent_dim,
                        "learning_rate": eta}
        
            torch.save(general_checkpoint, f"general_checkpoint_dcgan_like.pth")
    
    wandb.finish()
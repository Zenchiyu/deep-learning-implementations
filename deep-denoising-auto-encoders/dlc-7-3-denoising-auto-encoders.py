from typing import Any
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose, Lambda
from torchvision.transforms.functional import gaussian_blur

from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), embed_dim=16):
        super().__init__()
        C, H, W = input_shape
        self.conv1 = nn.Conv2d(C, 32, kernel_size=5)  # H -> H-5+1=H-4
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)  # H-4 -> H-4-5+1=H-8
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)  # H-8 -> (H-8-4)//2+1=(H-12)//2+1
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)  # (H-12)//2+1 -> ((H-12)//2+1-3)//2+1=((H-12)//2 - 2)//2+1
        # Up to here we have: N x 32 x ((H-12)//2 - 2)//2+1
        self.conv5 = nn.Conv2d(32, embed_dim, kernel_size=(((H-12)//2 - 2)//2+1,
                                                           ((W-12)//2 - 2)//2+1))
        # we get N x embed_dim x 1 x 1

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
        self.tconv4 = nn.ConvTranspose2d(32, 32, kernel_size=5)  # H-4 -> H-4-5+1=H-8
        self.tconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2)  # H-8 -> (H-8-4)//2+1=(H-12)//2+1
        self.tconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)  # (H-12)//2+1 -> ((H-12)//2+1-3)//2+1=((H-12)//2 - 2)//2+1
        # Up to here we have: N x 32 x ((H-12)//2 - 2)//2+1
        self.tconv1 = nn.ConvTranspose2d(embed_dim, 32, kernel_size=(((H-12)//2 - 2)//2+1,
                                                           ((W-12)//2 - 2)//2+1))
        # we go from N x embed_dim x 1 x 1 to N x *input_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No pooling
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = F.relu(self.tconv5(x))
        return x

class AutoEncoder(nn.Module):
    def __init__(self, input_shape=(1, 32, 32), embed_dim=16):
        super().__init__()
        self.encode = Encoder(input_shape, embed_dim)
        self.decode = Decoder(input_shape, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class Corruptor():
    def __call__(self, x: torch.Tensor, **kwds: Any) -> torch.Tensor:
        if "corrupt_type" in kwds:
            match kwds["corrupt_type"]:
                case "blur":  # Gaussian blur
                    k = kwds["kernel_size"]
                    sigma = kwds["sigma"]
                    return gaussian_blur(x, kernel_size=k, sigma=sigma)
                case "awgn":  # additive white gaussian noise
                    return x+x.new(x.shape).normal_(0, kwds["std"])
                case "erasure":  # pixel erasure
                    p = kwds["p"]  # proba erasure
                    mask = x.new(x.shape).bernoulli_(1-p)
                    return x*mask
                case "block":  # block masking
                    size = kwds["size"]
                    x_noisy = x.clone()
                    i, j = torch.randint()
                case _: # additive white gaussian noise
                    return x+x.new(x.shape).normal_(0, 0.1)
        else:
            return x+x.new(x.shape).normal_(0, kwds["std"])

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
    n_epochs = 100
    batch_size = 256
    dataset = CIFAR10
    dataset_name = dataset.__name__

    train_data, test_data = get_datasets(dataset=dataset, normalize_features=False)
    train_loader = DataLoader(train_data, num_workers=4, pin_memory=True,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=4, pin_memory=True,
                             batch_size=batch_size)

    N, (C, H, W) = len(train_loader), train_loader.dataset[0][0].shape

    # Model(s)
    model = AutoEncoder(input_shape=(C, H, W), embed_dim=16)
    model.to(device=device)
    # Number of parameters
    # sum(map(lambda x: x.numel(), model.parameters()))
    # or equivalently
    # sum([torch.prod(torch.tensor(p.size())) for p in model.parameters()])

    # Criterion
    criterion = nn.MSELoss()  # used for reconstruction loss
    criterion.to(device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Corruptor
    corruptor = Corruptor()

    for corrupt_type in ["awgn", "blur", "erasure"]:
        print("Corruption type:", corrupt_type)
        kwds = {"corrupt_type": corrupt_type}
        match corrupt_type:
            case "blur":  # Gaussian blur
                kwds["kernel_size"] = 5
                kwds["sigma"] = 4
            case "awgn":  # additive white gaussian noise
                kwds["std"] = 3
            case "erasure":  # pixel erasure
                kwds["p"] = 0.9  # proba to set some pixels to 0
            case _:
                kwds["std"] = 3
        # Training loop
        for e in tqdm(range(n_epochs)):
            acc_loss = 0
            for (X, _) in train_loader:  # ignore labels
                X = X.to(device=device)
                X_noisy = corruptor(X, **kwds)
                X_hat = model(X_noisy)
                loss = criterion(X_hat, X)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                acc_loss += loss.item()
            print(f"Epoch {e} acc_loss: {acc_loss}")

        with torch.no_grad():
            for (X, _) in test_loader:  # ignore labels
                X = X.to(device=device)
                X_noisy = corruptor(X, **kwds)
                X_hat = model(X_noisy)
                break

            n_rows, n_cols = 6, 12
            n_imgs = n_rows*n_cols
            torchvision.utils.save_image(X[:n_imgs], f"{dataset_name}_original_input_denoising_ae.png", 
                                        nrow=n_cols, pad_value=1.0)
            torchvision.utils.save_image(X_noisy[:n_imgs], f"{dataset_name}_noisy_input_denoising_ae_{corrupt_type}.png", 
                                        nrow=n_cols, pad_value=1.0)
            torchvision.utils.save_image(X_hat[:n_imgs], f"{dataset_name}_denoised_denoising_ae_{corrupt_type}.png", 
                                        nrow=n_cols, pad_value=1.0)
            
            fig, axs = plt.subplots(3, 1, figsize=(8, 2*n_rows))
            axs[0].imshow(plt.imread(f"{dataset_name}_original_input_denoising_ae.png"))
            axs[1].imshow(plt.imread(f"{dataset_name}_noisy_input_denoising_ae_{corrupt_type}.png"))
            axs[2].imshow(plt.imread(f"{dataset_name}_denoised_denoising_ae_{corrupt_type}.png"))
            axs[0].set_title(f"Original ({dataset_name})")
            axs[1].set_title(f"Corrupted/Noisy input using: {corrupt_type}")
            axs[2].set_title("Denoised/Reconstructed from denoising AE")
            
            for ax in axs:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(f"dlc-7-3-denoising-ae-{dataset_name}-{corrupt_type}.png")
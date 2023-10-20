<details>
  <summary>Denoising corrupted MNIST images with only $100$ epochs and $120657$ parameters</summary>

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/7f46699e-c534-4951-bfa4-9d322ce762d1" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/b93e3184-1dd8-4a58-9dba-b8a10e6ce897" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/0438ce7f-8eba-4113-9c03-da6f6ef0b1b3" width=300>
|:--:| :--:| :--:|

However, the training is unstable as the loss can stagnate from the first epoch.

Autoencoder with ReLU activation functions, no pooling, no interpolation:
```
AutoEncoder(
  (encode): Encoder(
    (conv1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
    (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
    (conv3): Conv2d(32, 32, kernel_size=(4, 4), stride=(2, 2))
    (conv4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2))
    (conv5): Conv2d(32, 16, kernel_size=(4, 4), stride=(1, 1))
  )
  (decode): Decoder(
    (tconv5): ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(1, 1))
    (tconv4): ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
    (tconv3): ConvTranspose2d(32, 32, kernel_size=(4, 4), stride=(2, 2))
    (tconv2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2))
    (tconv1): ConvTranspose2d(16, 32, kernel_size=(4, 4), stride=(1, 1))
  )
)
```


</details>

<details>
  <summary>Denoising corrupted CIFAR-10 images with only $100$ epochs and $133075$ parameters</summary>

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/bee4eebb-7d4c-4e20-9b57-2baf516edb59" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/a96879a6-a5df-4a6c-b961-a5118f7130a0" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/723dafd1-2c12-4527-a13e-2a0fb17ed667" width=300>
|:--:| :--:| :--:| 

However, the training is unstable as the loss can stagnate from the first epoch.

Autoencoder with ReLU activation functions, no pooling, no interpolation:
```
AutoEncoder(
  (encode): Encoder(
    (conv1): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1))
    (conv2): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
    (conv3): Conv2d(32, 32, kernel_size=(4, 4), stride=(2, 2))
    (conv4): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2))
    (conv5): Conv2d(32, 16, kernel_size=(5, 5), stride=(1, 1))
  )
  (decode): Decoder(
    (tconv5): ConvTranspose2d(32, 3, kernel_size=(5, 5), stride=(1, 1))
    (tconv4): ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(1, 1))
    (tconv3): ConvTranspose2d(32, 32, kernel_size=(4, 4), stride=(2, 2))
    (tconv2): ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2))
    (tconv1): ConvTranspose2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  )
)
```
</details>

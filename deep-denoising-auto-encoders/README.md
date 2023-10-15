Denoising corrupted images with only $100$ epochs and $133075$ parameters:

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/bee4eebb-7d4c-4e20-9b57-2baf516edb59" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/a96879a6-a5df-4a6c-b961-a5118f7130a0" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/723dafd1-2c12-4527-a13e-2a0fb17ed667" width=300>
|:--:| :--:| :--:| 

with only $100$ epochs.

Model with ReLU activation functions:
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
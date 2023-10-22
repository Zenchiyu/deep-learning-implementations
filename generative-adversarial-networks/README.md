Quick glimpse at generated samples!

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/ada5b7b7-ff81-4155-a630-01531082ff17" width=400> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/2db9c0fa-bf1d-4bcd-93f5-23e63f84b5be" width=400>
|:--:| :--:|
| GAN on test case | DCGAN-like on CIFAR-10 |


<details>
<summary>Standard Generative Adversarial Network on different test cases</summary>
  
I reimplemented the standard GAN from our course using Multi-Layer Perceptrons for both the generator and discriminator. We trained our model with 500 epochs in all plots except those on the right. These used 1000 epochs (about 3-4 min of training).

- First test case: A standard normal distribution split in two


| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/ada5b7b7-ff81-4155-a630-01531082ff17" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/a9516acb-30cb-4b85-ae87-3fcbfdb460f5" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/fecc0503-1541-4056-8082-16386603b129" width=300>
|:--:| :--:| :--:|

We created the data distribution by cutting a standard normal distribution in two and pushing the two parts by $1$ unit away from $0$.

- Second test case: A distribution in the form of a heart


| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/44b6915d-d616-4e60-81bf-c40d3a3d188c" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/f2cc00cf-7509-4396-a43c-48c38f24f6fd" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/d6b52332-49b9-4885-80f8-592420ca91fb" width=300>
|:--:| :--:| :--:|


Fake samples are in red, and real samples are in blue.

Generator:
```
Sequential(
  (0): Linear(in_features=16, out_features=100, bias=True)
  (1): ReLU()
  (2): Linear(in_features=100, out_features=2, bias=True)
)
```

Discriminator:
```
Sequential(
  (0): Linear(in_features=2, out_features=100, bias=True)
  (1): ReLU()
  (2): Linear(in_features=100, out_features=1, bias=True)
  (3): Sigmoid()
)
```

Remark(s): 
We train the generator to maximize the false acceptance of the discriminator. In contrast, we want the discriminator to correctly classify samples as real or fake by minimizing the binary cross entropy loss.

</details>

<details>
<summary>Deep Convolutional GAN-like on CIFAR-10</summary>

I reimplemented a Deep Convolutional GAN inspired by the DCGAN from our course using convolutional and batch normalization layers for both the generator and discriminator. Instead of using transposed convolutions in the generator, I used bilinear interpolations followed by "same convolutions" (keeping spatial resolution after conv.). Moreover, my generator is trained twice as much as the discriminator. I show some results I've got for different epochs.

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/c4b54565-4c2d-4b34-a63a-249b91201041" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/468a1c8d-6430-427b-adea-949f0c2d83f2" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/2db9c0fa-bf1d-4bcd-93f5-23e63f84b5be" width=300>
|:--:| :--:| :--:|
|After first epoch| After $6$ epochs | After $22$ epochs|

I've rescaled the generated samples to be in the [0, 1] range between displaying them. Initially, the generator seems to create edge detectors. As training progresses, some generated pictures look like boats while others don't make sense.

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/8ff75ef0-4b87-4028-9a0f-4897f75120a4" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/6d1d03b1-e9c5-4474-9f1a-2ece2f564bf5" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/56575292-5cc0-49f7-a679-c195b67fc639" width=300>
|:--:| :--:| :--:|
|After $32$ epochs| After $36$ epochs | After $50$ epochs|

We can try to observe our discriminator's performance for each epoch:

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/ad2acb15-1b2c-459c-9b0b-70b52ede8642" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/4b442c53-3a56-410a-a90e-7798a72427f2" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/f85b05fa-cd1a-4fe6-a2c9-2b2d017b5db5" width=300> 
|:--:| :--:| :--:|
|Average discriminator's output based on a batch of: real train, fake train , real test | Std discriminator's output based on a batch of: real train, fake train , real test | Generator and discriminator training losses based on a batch of real [x]or fake samples|

Remark(s): I forgot to create checkpoints and forgot to set the bias of the generator's last convolutional layer to false. I also forgot to set the generators and discriminators back to training mode after evaluation. It takes around $2$ hours to train my model with $50$ epochs on an AMD Ryzen 5 5600 6-Core Processor since I currently don't have access to a GPU. [The experiment was tracked using Weights & Biases](https://wandb.ai/stephane-nguyen/standard-gan-cnn/runs/7pxffg23?workspace=user-stephane-nguyen)



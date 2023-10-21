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

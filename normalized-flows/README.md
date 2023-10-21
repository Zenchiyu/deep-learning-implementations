<details>
<summary>Non-volume preserving network on different test cases</summary>

I reimplemented an NVP network (heavily based on Fleuret's course) using 4 invertible coupling layers made with Multi-Layer Perceptrons. We trained our model with 500 epochs in all plots except those on the right. These used 1000 epochs (about 5 min of training).

- First test case: A standard normal distribution split in two

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/7373b7a2-9dee-4c6f-bf4e-84ec1501c302" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/e7d16910-8e38-4c16-b072-ce48e1a3c7aa" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/a10b27aa-beef-442a-a7e9-fda80b13b996" width=300>
|:--:| :--:| :--:|

We created the data distribution by cutting a standard normal distribution in two and pushing the two parts by $1$ unit away from $0$.

- Second test case: A distribution in the form of a heart


| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/649cc04b-c915-4d00-8238-97b4b6c251a8" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/e901a850-d820-4538-9a8d-66c2bb235605" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/c21f6c7c-e1d6-4402-b5c2-e98ba320d06d" width=300>
|:--:| :--:| :--:|


In red, we have $f_\theta^{-1}(z)$ where $f_\theta$, our model, is a normalizing flow (transforming our data distribution into a standard multivariate normal distribution). The curves are obtained by applying either $f_\theta^{-1}$ (red)
or $f^{-1}$ (real, in blue) on circles.

Model:
```
NVPNet(
  (layers): ModuleList(
    (0-3): 4 x NVPCouplingLayer(
      (map_s): Sequential(
        (0): Linear(in_features=2, out_features=2, bias=True)
        (1): Tanh()
        (2): Linear(in_features=2, out_features=2, bias=True)
      )
      (map_t): Sequential(
        (0): Linear(in_features=2, out_features=2, bias=True)
        (1): Tanh()
        (2): Linear(in_features=2, out_features=2, bias=True)
      )
    )
  )
)
```

Remark(s):
We train our model by maximizing the log-likelihood (=minimizing the negative log-likelihood or empirical cross-entropy). This loss is tractable since we use [invertible] layers that simplify the computation of the determinant of the Jacobian in the formula of the change of variable in probability theory.


</details>

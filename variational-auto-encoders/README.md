- Variational Auto Encoders (VAE) (and $\beta$-VAE)

I've reimplemented a VAE based on two small convnets. The encoder predicts parameters of a gaussian distribution and the decoder predicts a conditional expectation (might be unlikely under the true data distribution).
In VAE, instead of optimizing the log-likelihood, we optimize the variational lower bound (ELBO: evidence lower bound). It tries to impose a distribution in the latent space (e.g. gaussian) and tries to have good reconstructions.

This is what I get with $300$ epochs for MNIST and CIFAR-10 (about 7-8 hours of training).

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/b3b0e513-805f-4fb6-a6b7-0b560dbb356a" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/2fa6ac5e-28a3-455a-8762-4fd70fd73e62" width=300> |
|:--:| :--:|

So there's definitely something wrong in my implementation or not enough capacity in my model. The loss is sometimes oscillating around a very high value from the first epoch!

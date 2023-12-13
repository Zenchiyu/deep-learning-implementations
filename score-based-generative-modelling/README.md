<details>
<summary>Interesting links/generalizations</summary>
  
- DDPM is related to SMLD since both minimize a weighted sum of denoising score matching objectives:
  - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) paper by Ho et al (2020) 
  - [What are diffusion models? Connection with NCSN (SMLD)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-noise-conditioned-score-networks-ncsn) blog post by Lilian Weng
- Stochastic Differential Equations (SDE) and Probability Flow ODE:
  - [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) paper by Song et al.:
    - DDPM and SMLD forward processes are special discretizations of different forward SDEs.
    - They encapsulate DDPM and SMLD samplers under their *predictor-corrector samplers*: DDPM sampler only has a "predictor" (reverse process is a special discretization of its reverse-time SDE) while SMLD only has a "corrector" (annealed Langevin dynamics)
    - Deterministic sampling by discretizing the probability flow ODE
  - [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) paper by Karras et al.:
    - Rewrite the Probability Flow ODE of Song et al. into a more intuitive equation
    - Same for the forward/reverse-time SDEs
    - Implicit Langevin term in SDEs
    - Euler/Heun + explicit Langevin-like stochastic part -> stochastic sampling. No SDE solver
    - Input, output normalizations. Also handle tradeoff between predicting the noise or the original image at different noise levels (don't want model to just collapse to the identity function)
      
  So we can sample pictures by explicitly using numerical ODE/SDE solvers (or variants) after estimating the score functions of perturbed data distributions $p_t(x)$ or $p(x, \sigma(t))$. There's a connection between the Probability Flow ODE and SDE. Note that [Karras et al.](https://arxiv.org/abs/2206.00364) differentiate $p_t(x)$ with $p(x, \sigma(t))$. The latter is obtained only through a convolution between the data distribution and a gaussian distribution (See Appendix B, p.21).
</details>

<details>
<summary>Score matching with Langeving Dynamics on my toy distributions</summary>

I've reimplemented a Score matching with Langeving Dynamics (SMLD) based on the "[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)" paper by Yang Song and Stefano Ermon. My Noise Conditional Score Network (NCSN) is a $3$-layer MLP with soft plus activation functions since the authors also used a similar architecture on their toy examples.


We trained our model with 1000 epochs (about 7-8 min of training). The first column gives vector fields corresponding to the estimated score functions for perturbed data distributions with $\sigma=0.01$. The second column shows generated samples in red and real samples in blue. The last column also shows a partial trajectory by the Annealed Langevin Dynamics (ignoring the first 250 steps). The score of a distribution $p(x)$ is $\nabla_x \log p(x)$

- First test case: A standard normal distribution split in two

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/bc24135e-1621-41e5-846a-647311c7fbe1" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/65fe4c76-acef-467b-add1-4fa795aacf5f" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/6b4310d7-caa4-4d7b-907b-36040414b5d5" width=300>
|:--:| :--:| :--:|

We created the data distribution by cutting a standard normal distribution in two and pushing the two parts by $1$ unit away from $0$.

- Second test case: A distribution in the form of a heart


| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/b15b39d7-4ca1-4747-9e39-d327b4baab44" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/47687903-f9f0-4bea-a0fb-90bc2c32bea7" width=300> | <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/0965b333-cf0c-4f9d-83f3-f31f16748b6f" width=300>
|:--:| :--:| :--:|

Due to the annealing process (noise level reduction), we can observe that the trajectory becomes less erratic as it continues.


Model:
```
NoiseConditionalScoreNetwork(
  (layers): Sequential(
    (0): Linear(in_features=3, out_features=128, bias=True)
    (1): Softplus(beta=1, threshold=20)
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): Softplus(beta=1, threshold=20)
    (4): Linear(in_features=128, out_features=2, bias=True)
  )
)
```
The additional input feature corresponds to the standard deviation $\sigma$ in $s_\theta(x, \sigma)$.

Remark(s): We don't maximize the log-likelihood (e.g. in NVP), a surrogate such as the evidence lower bound (see VAE), or train models in an adversarial setting (e.g. GAN). Instead, the SMLD's training consists of estimating the score function of the data distribution and then using it to get samples at inference time (via the annealed Langevin dynamics, inspired by Simulated Annealing).

"Key sentences":
- Aggregating individual denoising score matching objectives 
- Score matching
- Annealed Langevin dynamics for sampling
</details>

<details>
<summary>Denoising Diffusion Probabilistic Model using an MLP on my toy distributions</summary>

I've reimplemented a Denoising Diffusion Probabilistic Model (DDPM) based on the "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)" paper by Ho et al (2020). I use a network similar to `NoiseConditionalScoreNetwork` to predict the noise. I condition the model on a scaled version of the time instead of noise (the scaling matters a lot, w/o => cannot sample).

We trained our model with 1000 epochs (about 12-13 min of training). We display generated samples in red and real samples in blue. We also show in black a partial trajectory by the reverse process (ignoring the first 250 steps).

- First test case: A standard normal distribution split in two

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/0aafe0b0-3468-4c98-9c2e-b3c919fe3ead" width=300>
|:--:|

We created the data distribution by cutting a standard normal distribution in two and pushing the two parts by $1$ unit away from $0$.

- Second test case: A distribution in the form of a heart

| <img src="https://github.com/Zenchiyu/deep-learning-implementations/assets/49496107/1423b55a-03fc-4e59-b84c-1af04776f856" width=300>
|:--:|


Model:
```
NoisePredictor(
  (layers): Sequential(
    (0): Linear(in_features=3, out_features=128, bias=True)
    (1): Softplus(beta=1, threshold=20)
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): Softplus(beta=1, threshold=20)
    (4): Linear(in_features=128, out_features=2, bias=True)
  )
)
```

I condition the model on a scaled version of the time instead of noise (the scaling matters a lot, w/o => cannot sample).

Remark(s): DDPM indirectly estimates the parameters of the conditional disitributions (gaussians) in the reverse process
</details>

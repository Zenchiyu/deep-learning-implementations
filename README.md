# Deep learning implementations

Deep learning implementations

Note: For toy distributions I generally don't normalize my data.

## Ramping-up

In this section, I explain what are some of the concepts (non-exhaustive) I find very useful in my ramp-up when going through Fleuret's course, paper(s) and other resources.


<details>
  <summary>I've implemented and learned concepts in the following order</summary>
  
- A CNN from scratch (convolution and transposed convolutions from scratch) (although my backpropagation from scratch didn't work due to tensor shape mismatches)
- Maximum (log)-likelihood, Maximum A Posterior, Cross-Entropy Loss
- Deep Auto-Encoder (AE) using convnets (with transposed convolutions for the decoder)
- Deep Denoising AE
- Variational AE (although it didn't really give good results)
- Non-Volume Preserving Networks with coupling layers
- Generative Adversarial Networks (although it didn't really give good results)
- Noise Conditional Score Networks (on my toy distributions)
- Denoising Diffusion Probabilistic Model (on my toy distributions)
- EDM diffusion model based on [the paper](https://arxiv.org/abs/2206.00364) by Karras et al. (currently in a private repository, can generate unconditionally and conditionally for FashionMNIST and CIFAR-10 with Classifier-Free Guidance. Can generate also faces from CelebA.). The model is a U-net with residual blocks and multi-head self attention blocks (self-attention in lower resolution blocks.)
</details>

<details>
  <summary>Work In Progress/TODO</summary>

- Noise Conditional Score Networks on CIFAR-10 or FashionMNIST (not working yet so I didn't upload it)
- Denoising Diffusion Probabilistic Model (not working yet so I didn't upload it)
- Generative Pre-trained Transformer, decoder-only part of the Transformer (not completely working, issues with long-term dependencies)
- TODO: Latent Diffusion Model by using a diffusion model in the latent space. Encoder and decoder from VQ-VAE or VQ-GAN.
- TODO: Video diffusion (based on GAIA-1 and TECO) where the dynamics & temporal consistency come from a transformer and realistic/detailed videos come from the diffusion model. The transformer would be working on tokens obtained from the Vector-quantization and the diffusion model would be generating frames conditionned on generated tokens (by AR transformer). Note: from the encoder-decoder pair, the diffusion model replaces the decoder. 
</details>

<details>
  <summary>Other useful concepts</summary>
  
- Information theoretical concepts:
  - Entropy
  - Cross-entropy
  - Mutual Information
  - Kullback-Leibler divergence
- Metaheuristics for optimization concepts:
  - Particle Swarm Optimization (interesting relation to the Momentum optimization method)
  - Simulated Annealing (interesting relation to the Noise Conditional Score Network)
  - (Genetic algorithms)
- Modelisation and simulation of natural phenomena concepts:
  - Monte-Carlo Markov Chain
  - Diffusion Process
- Other:
  - Importance-Sampling
  - Moving averages (incl. Exponentially Weighted Moving Averages)

  
</details>

# Credits
- https://fleuret.org/dlc/
- https://fleuret.org/francois/lbdl.html

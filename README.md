# Deep learning implementations

Deep learning implementations.

## Step-by-step

In this section, I explain what are some of the concepts (non-exhaustive) I find very useful when going through Fleuret's course and paper(s).


<details>
  <summary>I've implemented and learned concepts in the following order</summary>
  
- A CNN from scratch (convolution and transposed convolutions from scratch) (although my backpropagation from scratch didn't work due to tensor shape mismatches)
- Maximum (log)-likelihood, Maximum A Posterior, Cross-Entropy Loss
- Deep Auto-Encoder (AE) using convnets (with transposed convolutions for the decoder)
- Deep Denoising AE
- Variational AE (although it didn't really give good results)
- Non-Volume Preserving Networks with coupling layers
- Generative Adversarial Networks (although it didn't really give good results)
- Noise Conditional Score Networks
</details>


<details>
  <summary>Other useful concepts</summary>
  
- Information theoretical concepts:
  - Entropy
  - Cross-emtropy
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

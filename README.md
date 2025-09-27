# Variational Autoencoder (VAE) on Fashion-MNIST

This project implements an enhanced Variational Autoencoder (VAE) in PyTorch, with separate Encoder and Decoder modules.
Unlike a traditional Autoencoder, the VAE models the latent space as a probabilistic distribution and uses the reparameterization trick for sampling.
Key enhancements:

β-VAE objective: balances reconstruction accuracy and latent space disentanglement using a tunable β factor.

Convolutional Encoder/Decoder: improves representation learning for image data compared to fully-connected networks.

2D latent embedding space: allows visualization of how the model clusters different Fashion-MNIST classes.

Sampling & generation: the trained decoder can generate new clothing images by sampling from the latent distribution.

The VAE is trained on the Fashion-MNIST dataset to both reconstruct input images and generate novel samples of clothing.

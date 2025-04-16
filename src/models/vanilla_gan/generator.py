"""Vanilla Generator for the original GAN implementation."""

import numpy as np
from torch import nn
from src.models.base.base_generator import BaseGenerator


class VanillaGenerator(BaseGenerator):
    """
    Generator network for the original GAN.
    Uses fully connected layers to transform latent vectors into images.
    """

    def __init__(self, latent_dim, hidden_dim, img_shape):
        """
        Initialize the generator.

        Args:
            latent_dim: Dimension of the latent space
            hidden_dim: Dimension of hidden layers
            img_shape: Shape of the output image (channels, height, width)
        """
        super().__init__({"model": {"latent_dim": latent_dim}})

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.img_shape = img_shape

        # Calculate total number of pixels in the image
        self.img_size = int(np.prod(img_shape))

        # Build the network
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Third hidden layer
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer
            nn.Linear(hidden_dim * 4, self.img_size),
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z: Latent vector of shape [batch_size, latent_dim]

        Returns:
            Generated images of shape [batch_size, C, H, W]
        """
        # Generate flattened images
        img_flat = self.model(z)

        # Reshape to proper image dimensions
        img = img_flat.view(img_flat.size(0), *self.img_shape)

        return img

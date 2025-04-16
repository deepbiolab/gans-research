"""
This module defines base generator class for GAN implementations.
"""

from abc import ABC, abstractmethod
from torch import nn


class BaseGenerator(nn.Module, ABC):
    """
    Base class for all generator implementations.
    This provides a common interface for different generator architectures.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["model"]["latent_dim"]

    @abstractmethod
    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z: Latent vector of shape [batch_size, latent_dim]

        Returns:
            Generated images
        """

    def init_weights(self, module):
        """Initialize the weights of the network for better training."""
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

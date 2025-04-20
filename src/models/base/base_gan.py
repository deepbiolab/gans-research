"""
BaseGAN module provides the base class for implementing different GAN variants.
"""
from abc import ABC, abstractmethod
import torch
from torch import nn


class BaseGAN(nn.Module, ABC):
    """
    Base class for all GAN implementations.
    This provides a common interface for different GAN variants.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["model"]["latent_dim"]
        self.device = torch.device(config["experiment"]["device"])

        # These will be implemented by specific GAN variants
        self.generator = None
        self.discriminator = None

    @abstractmethod
    def build_generator(self):
        """Build the generator network."""

    @abstractmethod
    def build_discriminator(self):
        """Build the discriminator/critic network."""

    def generate_latent(self, batch_size):
        """Generate random latent vectors."""
        return torch.randn(batch_size, self.latent_dim).to(self.device)

    def generate_images(self, batch_size=None, latent_vectors=None):
        """Generate images from random latent vectors or provided ones."""
        if latent_vectors is None:
            if batch_size is None:
                raise ValueError("Either batch_size or latent_vectors must be provided")
            latent_vectors = self.generate_latent(batch_size)

        with torch.no_grad():
            images = self.generator(latent_vectors)

        return images

    @classmethod
    def from_pretrained(cls, path, map_location=None, **kwargs):
        """Load a pre-trained model from a checkpoint."""
        checkpoint = torch.load(path, map_location=map_location)
        config = kwargs.get("config", checkpoint.get("config", {}))
        
        # Create new model instance
        model = cls(config)
        model.build_generator()
        model.build_discriminator()
        
        # Loading weights
        model.generator.load_state_dict(checkpoint["generator_state_dict"])
        model.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        
        return model

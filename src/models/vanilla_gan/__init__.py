"""
Implementation of the original GAN as proposed by Goodfellow et al. in 2014.
Uses fully connected layers for both generator and discriminator.
"""

# Import the base GAN class
from src.models.base.base_gan import BaseGAN
from .generator import VanillaGenerator
from .discriminator import VanillaDiscriminator


class VanillaGAN(BaseGAN):
    """
    Implementation of the original GAN as proposed by Goodfellow et al. in 2014.
    Uses fully connected layers for both generator and discriminator.
    """

    def __init__(self, config):
        super().__init__(config)

        self.latent_dim = config["model"]["latent_dim"]
        self.hidden_dim = config["model"]["hidden_dim"]
        self.img_shape = (
            config["data"]["channels"],
            config["data"]["image_size"],
            config["data"]["image_size"],
        )

        # Build the generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Initialize weights
        self.generator.apply(self.generator.init_weights)
        self.discriminator.apply(self.discriminator.init_weights)

        # Move models to the specified device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def build_generator(self):
        """Build the generator network."""
        return VanillaGenerator(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            img_shape=self.img_shape,
        )

    def build_discriminator(self):
        """Build the discriminator network."""
        return VanillaDiscriminator(
            img_shape=self.img_shape, hidden_dim=self.hidden_dim
        )


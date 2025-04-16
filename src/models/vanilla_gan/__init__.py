"""
Implementation of the original GAN as proposed by Goodfellow et al. in 2014.
Uses fully connected layers for both generator and discriminator.
"""

# Import torch for tensor operations
import torch
from torch import nn

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

    def generator_loss(self, fake_pred, *args, **kwargs):
        """Calculate the loss for the generator."""
        # We want the discriminator to classify fake images as real (1)
        target_real = torch.ones_like(fake_pred, device=self.device)
        return nn.BCEWithLogitsLoss()(fake_pred, target_real)

    def discriminator_loss(self, real_pred, fake_pred, *args, **kwargs):
        """Calculate the loss for the discriminator."""
        # Real images should be classified as real (1)
        target_real = torch.ones_like(real_pred, device=self.device)
        # Fake images should be classified as fake (0)
        target_fake = torch.zeros_like(fake_pred, device=self.device)

        # Calculate binary cross entropy loss
        real_loss = nn.BCEWithLogitsLoss()(real_pred, target_real)
        fake_loss = nn.BCEWithLogitsLoss()(fake_pred, target_fake)

        # Total discriminator loss is the average of real and fake losses
        return (real_loss + fake_loss) / 2.0

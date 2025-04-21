"""
Implementation of the Deep Convolutional GAN (DCGAN) as proposed by Radford et al. in 2015.
Uses deep convolutional layers for both generator and discriminator.
"""

from typing import Any, Dict
from torch import nn

from src.models.base.base_gan import BaseGAN
from .generator import DCGANGenerator
from .discriminator import DCGANDiscriminator


class DCGAN(BaseGAN):
    """
    Deep Convolutional GAN (DCGAN) implementation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.latent_dim: int = config["model"]["latent_dim"]
        self.ngf: int = config["model"]["ngf"]
        self.ndf: int = config["model"]["ndf"]
        self.use_batch_norm: bool = config["model"].get("use_batch_norm", True)
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

    def build_generator(self) -> nn.Module:
        """
        Build the DCGAN generator network.
        """
        return DCGANGenerator(
            latent_dim=self.latent_dim,
            ngf=self.ngf,
            nc=self.img_shape[0],
            image_size=self.img_shape[1],
            use_batch_norm=self.use_batch_norm,
        )

    def build_discriminator(self) -> nn.Module:
        """
        Build the DCGAN discriminator network.
        """
        return DCGANDiscriminator(
            ndf=self.ndf,
            nc=self.img_shape[0],
            image_size=self.img_shape[1],
            use_batch_norm=self.use_batch_norm,
        )

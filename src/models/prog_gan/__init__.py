"""
Implementation of the Progressive Growing GAN (ProGAN) as proposed by Karras et al. in 2018.
Supports progressive resolution growth and fade-in blending.
"""

from typing import Any, Dict
import torch
from torch import nn

from src.models.base.base_gan import BaseGAN
from .generator import Generator
from .discriminator import Discriminator


class ProgGAN(BaseGAN):
    """
    Progressive Growing GAN (ProGAN) implementation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.latent_dim: int = config["model"]["latent_dim"]
        self.max_resolution: int = config["data"].get("image_size", 512)
        self.img_channels: int = config["data"]["channels"]
        self.fmap_base: int = config["model"].get("feature_maps", 8192)
        self.fmap_decay: float = config["model"].get("fmap_decay", 1.0)
        self.fmap_max: int = config["model"].get("fmap_max", 512)

        # Progressive growing parameters
        self.current_res: int = config["model"].get("starting_resolution", 4)
        self.alpha: float = 1.0  # blending coefficient for fade-in

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
        Build the ProGAN generator network.
        """
        return Generator(self.config)

    def build_discriminator(self) -> nn.Module:
        """
        Build the ProGAN discriminator network.
        """
        return Discriminator(self.config)

    def generate_images(
        self, batch_size=None, latent_vectors=None, current_res=None, alpha=None
    ):
        """
        Generate images at the current progressive resolution.
        """
        if latent_vectors is None:
            if batch_size is None:
                raise ValueError("Either batch_size or latent_vectors must be provided")
            latent_vectors = self.generate_latent(batch_size)

        if current_res is None:
            current_res = self.current_res
        if alpha is None:
            alpha = self.alpha

        with torch.no_grad():
            images = self.generator(
                latent_vectors, current_res=current_res, alpha=alpha
            )
        return images

"""
Implementation of the Wasserstein GAN with Gradient Penalty (WGAN-GP) as proposed by
Gulrajani et al. in "Improved Training of Wasserstein GANs".

WGAN-GP improves upon WGAN by replacing weight clipping with a gradient penalty term
to enforce the Lipschitz constraint on the critic, resulting in more stable training
and better quality results.
"""

from typing import Any, Dict
from torch import nn

from src.models.base.base_gan import BaseGAN
from .generator import WGANGPGenerator
from .discriminator import WGANGPDiscriminator


class WGANGP(BaseGAN):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation.

    This model extends the original WGAN by replacing weight clipping with a gradient
    penalty to enforce the Lipschitz constraint on the critic. The gradient penalty
    ensures that the gradient norm of the critic's output with respect to its inputs
    is close to 1 at interpolated points between real and fake samples.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.latent_dim: int = config["model"]["latent_dim"]
        self.ngf: int = config["model"]["ngf"]
        self.ndf: int = config["model"]["ndf"]
        self.use_batch_norm: bool = config["model"].get("use_batch_norm", True)
        self.lambda_gp: float = config["training"].get("lambda_gp", 10.0)
        self.img_shape = (
            config["data"]["channels"],
            config["data"]["image_size"],
            config["data"]["image_size"],
        )

        # Build the generator and critic (discriminator)
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
        Build the WGAN-GP generator network.
        """
        return WGANGPGenerator(
            latent_dim=self.latent_dim,
            ngf=self.ngf,
            nc=self.img_shape[0],
            image_size=self.img_shape[1],
            use_batch_norm=self.use_batch_norm,
        )

    def build_discriminator(self) -> nn.Module:
        """
        Build the WGAN-GP critic network.
        Unlike the original WGAN, the WGAN-GP critic doesn't use weight clipping.
        """
        return WGANGPDiscriminator(
            ndf=self.ndf,
            nc=self.img_shape[0],
            image_size=self.img_shape[1],
            use_batch_norm=self.use_batch_norm,
        )

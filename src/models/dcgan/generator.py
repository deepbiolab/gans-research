"""
DCGAN Generator implementation.
"""

from typing import Dict, Any
import torch
from torch import nn
from src.models.base.base_generator import BaseGenerator


class Block(nn.Module):
    """
    Basic block for the DCGAN generator.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_bn: bool,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.
        """
        return self.block(x)


class DCGANGenerator(BaseGenerator):
    """
    Deep Convolutional GAN (DCGAN) Generator.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        ngf: int = 64,
        nc: int = 3,
        image_size: int = 64,
        use_batch_norm: bool = True,
        config: Dict[str, Any] = None,
    ) -> None:
        """
        Args:
            latent_dim (int): Dimension of the latent input vector.
            ngf (int): Number of generator feature maps.
            nc (int): Number of output channels (e.g., 3 for RGB).
            image_size (int): Output image size (assumes square images).
            use_batch_norm (bool): Whether to use BatchNorm in layers.
            config (dict): Optional config dict passed to BaseGenerator.
        """
        if config is None:
            config = {"model": {"latent_dim": latent_dim}}
        super().__init__(config)

        assert image_size == 64, "This implementation only supports 64x64 images."

        self.latent_dim = latent_dim
        self.ngf = ngf
        self.nc = nc
        self.image_size = image_size
        self.use_batch_norm = use_batch_norm

        self.net = nn.Sequential(
            # Input: latent_dim x 1 x 1
            Block(latent_dim, ngf * 8, 4, 1, 0, use_batch_norm),  # (ngf*8) x 4 x 4
            Block(ngf * 8, ngf * 4, 4, 2, 1, use_batch_norm),     # (ngf*4) x 8 x 8
            Block(ngf * 4, ngf * 2, 4, 2, 1, use_batch_norm),     # (ngf*2) x 16 x 16
            Block(ngf * 2, ngf, 4, 2, 1, use_batch_norm),         # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),           # (nc) x 64 x 64
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator.

        Args:
            z (torch.Tensor): Input latent vector of shape (N, latent_dim, 1, 1).
        Returns:
            torch.Tensor: Generated image tensor of shape (N, nc, image_size, image_size).
        """
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)  # (N, latent_dim, 1, 1)
        return self.net(z)

"""
DCGAN Discriminator implementation.
"""

from typing import Dict, Any
import torch
from torch import nn
from src.models.base.base_discriminator import BaseDiscriminator


class Block(nn.Module):
    """
    Basic block for the DCGAN discriminator.
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
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.
        """
        return self.block(x)


class DCGANDiscriminator(BaseDiscriminator):
    """
    Deep Convolutional GAN (DCGAN) Discriminator.
    """

    def __init__(
        self,
        ndf: int = 64,
        nc: int = 3,
        image_size: int = 64,
        use_batch_norm: bool = True,
        config: Dict[str, Any] = None,
    ) -> None:
        """
        Args:
            ndf (int): Number of discriminator feature maps.
            nc (int): Number of input channels (e.g., 3 for RGB).
            image_size (int): Input image size (assumes square images).
            use_batch_norm (bool): Whether to use BatchNorm in layers (not used in the first layer).
            config (dict): Optional config dict passed to BaseDiscriminator.
        """
        if config is None:
            config = {"model": {}}
        super().__init__(config)

        assert image_size == 64, "This implementation only supports 64x64 images."

        self.ndf = ndf
        self.nc = nc
        self.image_size = image_size
        self.use_batch_norm = use_batch_norm

        self.net = nn.Sequential(
            # Input: (nc) x 64 x 64
            Block(nc, ndf, 4, 2, 1, False),                      # (ndf) x 32 x 32
            Block(ndf, ndf * 2, 4, 2, 1, use_batch_norm),         # (ndf*2) x 16 x 16
            Block(ndf * 2, ndf * 4, 4, 2, 1, use_batch_norm),     # (ndf*4) x 8 x 8
            Block(ndf * 4, ndf * 8, 4, 2, 1, use_batch_norm),     # (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),           # 1 x 1 x 1
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, nc, image_size, image_size).
        Returns:
            torch.Tensor: Probability tensor of shape (N, 1, 1, 1).
        """
        return self.net(img)

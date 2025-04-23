"""
Progressive growing generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.base.base_generator import BaseGenerator


def num_filters(
    stage: int, fmap_base: int = 8192, fmap_decay: float = 1.0, fmap_max: int = 512
) -> int:
    """
    A small helper function to compute the number of filters for conv layers based on the depth.
    From the original repo https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L252

    Args:
        stage: Current resolution stage (higher stage = lower resolution)
        fmap_base: Base filter count for the network
        fmap_decay: Controls how quickly filters are reduced at higher resolutions
        fmap_max: Maximum number of filters allowed

    Returns:
        Number of filters to use for the given stage
    """
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


class GeneratorFirstBlock(nn.Module):
    """
    First block for ProGAN Generator: latent vector -> feature map (4x4).

    This initial block transforms the latent vector into the smallest spatial feature map,
    which will be progressively grown in subsequent blocks.

    Args:
        latent_dim: Dimension of the input latent vector
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        # Create a sequence of operations to transform latent vector to 4x4 feature map
        self.block = nn.Sequential(
            # Transposed convolution to convert 1x1 to 4x4 spatial dimensions
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4),
            # Activation function
            nn.LeakyReLU(0.2),
            # Additional convolution to refine features
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # Activation function
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the first generator block.

        Args:
            x: Input tensor of shape (B, latent_dim)

        Returns:
            Tensor of shape (B, 512, 4, 4)
        """
        # Add spatial dimensions to the latent vector
        x = x[..., None, None]
        # Apply convolutional block to get 4x4 feature map
        x = self.block(x)
        return x


class GeneratorBlock(nn.Module):
    """
    Progressive block for ProGAN Generator: upsample + convs.

    Each block doubles the spatial resolution of the feature map.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Convolutional block that processes the upsampled feature map
        self.block = nn.Sequential(
            # First convolution to process upsampled features
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # Activation function
            nn.LeakyReLU(0.2),
            # Second convolution to refine features
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            # Activation function
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a generator block.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Tensor of shape (B, out_channels, H*2, W*2)
        """
        # Upsample the feature map to double the spatial dimensions
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolutional block
        x = self.block(x)
        return x


class Generator(BaseGenerator):
    """
    ProGAN Generator: progressively grows from 4x4 to max_resolution.

    The generator starts with a small resolution and progressively adds layers
    to increase the output resolution. During training, new layers are smoothly
    faded in using the alpha parameter.

    Args:
        max_resolution: Maximum output resolution (must be a power of 2)
        latent_dim: Dimension of the input latent vector
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.max_resolution = config["data"].get("image_size", 512)
        self.latent_dim = config["model"]["latent_dim"]
        self.img_channels = config["data"]["channels"]
        self.fmap_base = config["model"].get("feature_maps", 8192)
        self.fmap_decay = config["model"].get("fmap_decay", 1.0)
        self.fmap_max = config["model"].get("fmap_max", 512)

        # Calculate all resolutions from 4x4 up to max_resolution
        self.resolutions = [
            2**i for i in range(2, int(np.log2(self.max_resolution)) + 1)
        ]
        # Container for generator blocks
        self.blocks = nn.ModuleList()
        # Container for to_rgb conversion layers
        self.to_rgb = nn.ModuleList()

        # Initialize the first block (4x4)
        self.blocks.append(GeneratorFirstBlock(self.latent_dim))
        # RGB conversion for the first resolution
        self.to_rgb.append(
            nn.Conv2d(
                num_filters(1, self.fmap_base, self.fmap_decay, self.fmap_max),
                self.img_channels,
                kernel_size=1,
            )
        )

        # Initialize progressive blocks for higher resolutions
        for stage in range(1, len(self.resolutions)):
            self.blocks.append(
                GeneratorBlock(
                    num_filters(stage, self.fmap_base, self.fmap_decay, self.fmap_max),
                    num_filters(
                        stage + 1, self.fmap_base, self.fmap_decay, self.fmap_max
                    ),
                )
            )
            self.to_rgb.append(
                nn.Conv2d(
                    num_filters(
                        stage + 1, self.fmap_base, self.fmap_decay, self.fmap_max
                    ),
                    self.img_channels,
                    kernel_size=1,
                )
            )

    def forward(
        self, z: torch.Tensor, current_res: int, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass for the generator.

        Args:
            z: Latent vector of shape (B, latent_dim)
            current_res: Target output resolution (must be in self.resolutions)
            alpha: Fade-in blending coefficient (1.0 = only new block, 0.0 = only previous block)

        Returns:
            RGB image tensor of shape (B, 3, current_res, current_res)
        """
        # Find the index of the current resolution
        res_idx = self.resolutions.index(current_res)

        # Process through the first block
        x = self.blocks[0](z)

        # If we're at the lowest resolution, convert directly to RGB
        if res_idx == 0:
            return self.to_rgb[0](x)

        # Process through intermediate blocks
        for i in range(1, res_idx):
            x = self.blocks[i](x)

        # Generate the RGB image at the previous resolution
        prev_img = self.to_rgb[res_idx - 1](x)
        # Upsample to match the current resolution
        prev_img = F.interpolate(prev_img, scale_factor=2, mode="nearest")

        # Process through the final block for current resolution
        x = self.blocks[res_idx](x)
        # Generate the RGB image at the current resolution
        new_img = self.to_rgb[res_idx](x)

        # Blend between the two images based on alpha
        return new_img * alpha + prev_img * (1 - alpha)

"""
Progressive growing discriminator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.base.base_discriminator import BaseDiscriminator


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


class DiscriminatorLastBlock(nn.Module):
    """
    Last block for ProGAN Discriminator: (4x4) feature map → scalar.

    This final block processes the smallest 4x4 feature map and outputs
    a single scalar value representing the discriminator's prediction.

    Args:
        in_channels: Number of input channels
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Create sequential block to process 4x4 feature map to scalar output
        self.block = nn.Sequential(
            # First convolution to process features
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # Activation function
            nn.LeakyReLU(0.2),
            # Final convolution that reduces spatial dimensions to 1x1
            nn.Conv2d(in_channels, 1, kernel_size=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the last discriminator block.

        Args:
            x: Input tensor of shape (B, in_channels, 4, 4)

        Returns:
            Tensor of shape (B, 1) containing the discriminator scores
        """
        # Apply convolutional block
        x = self.block(x)
        # Flatten the output to (B, 1)
        return x.view(x.size(0), -1)


class DiscriminatorBlock(nn.Module):
    """
    Progressive block for ProGAN Discriminator: convs + downsample.

    Each block halves the spatial resolution of the feature map.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Convolutional block that processes features before downsampling
        self.block = nn.Sequential(
            # First convolution to process input features
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            # Activation function
            nn.LeakyReLU(0.2),
            # Second convolution to change channel dimensions
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # Activation function
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a discriminator block.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Tensor of shape (B, out_channels, H/2, W/2)
        """
        # Apply convolutional block
        x = self.block(x)
        # Downsample the feature map to halve the spatial dimensions
        x = F.avg_pool2d(x, kernel_size=2)
        return x


class Discriminator(BaseDiscriminator):
    """
    ProGAN Discriminator: progressively shrinks from max_resolution to 4x4.

    The discriminator starts with a high resolution input and progressively
    reduces it through a series of blocks. During training, new layers are
    smoothly faded in using the alpha parameter.

    Args:
        max_resolution: Maximum input resolution (must be a power of 2)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.max_resolution = config["data"].get("image_size", 512)
        self.img_channels = config["data"]["channels"]
        self.fmap_base = config["model"].get("feature_maps", 8192)
        self.fmap_decay = config["model"].get("fmap_decay", 1.0)
        self.fmap_max = config["model"].get("fmap_max", 512)

        # Calculate all resolutions from 4x4 up to max_resolution
        self.resolutions = [
            2**i for i in range(2, int(np.log2(self.max_resolution)) + 1)
        ]
        # Container for discriminator blocks
        self.blocks = nn.ModuleList()
        # Container for from_rgb conversion layers
        self.from_rgb = nn.ModuleList()

        # Initialize blocks: highest resolution first
        for stage in reversed(range(1, len(self.resolutions))):
            # Add a new block that halves the resolution
            self.blocks.append(
                DiscriminatorBlock(
                    num_filters(
                        stage + 1, self.fmap_base, self.fmap_decay, self.fmap_max
                    ),
                    num_filters(stage, self.fmap_base, self.fmap_decay, self.fmap_max),
                )
            )
            # Add RGB conversion for this resolution
            self.from_rgb.append(
                nn.Conv2d(
                    self.img_channels,
                    num_filters(
                        stage + 1, self.fmap_base, self.fmap_decay, self.fmap_max
                    ),
                    kernel_size=1,
                )
            )

        # Initialize the last block (4x4)
        self.last_block = DiscriminatorLastBlock(
            num_filters(1, self.fmap_base, self.fmap_decay, self.fmap_max)
        )
        # RGB conversion for the lowest resolution (4x4)
        self.from_rgb.append(
            nn.Conv2d(
                self.img_channels,
                num_filters(1, self.fmap_base, self.fmap_decay, self.fmap_max),
                kernel_size=1,
            )
        )

    def forward(
        self, img: torch.Tensor, current_res: int, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass for the discriminator.

        Args:
            x: Input image of shape (B, 3, current_res, current_res)
            current_res: Input resolution (must be in self.resolutions)
            alpha: Fade-in blending coefficient (1.0 = only new block, 0.0 = only previous block)

        Returns:
            Tensor of shape (B, 1) containing the discriminator scores
        """
        # Find the index of the current resolution
        res_idx = self.resolutions.index(current_res)

        block_idx = len(self.resolutions) - res_idx - 1

        # If we're at the lowest resolution (4x4), use only the last block
        if res_idx == 0:
            # Convert RGB to features
            out = self.from_rgb[-1](img)
            # Process through the last block
            out = self.last_block(out)
            return out

        # For higher resolutions, implement fade-in mechanism

        # Process input at current resolution
        high_in = self.from_rgb[block_idx](img)
        high_out = self.blocks[block_idx](high_in)

        # Process downsampled input at previous resolution
        x_down = F.avg_pool2d(img, kernel_size=2)
        low_in = self.from_rgb[block_idx + 1](x_down)

        # Blend the outputs based on alpha
        out = high_out * alpha + low_in * (1 - alpha)

        # Process through remaining blocks
        for i in range(block_idx + 1, len(self.blocks)):
            out = self.blocks[i](out)

        # Process through the last block
        out = self.last_block(out)
        return out

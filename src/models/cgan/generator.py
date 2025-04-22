"""
Conditional Generator for CGAN (MNIST, fully connected, one-hot label conditioning)
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.models.base.base_generator import BaseGenerator

class CGANGenerator(BaseGenerator):
    """
    Generator network for Conditional GAN.
    Combines latent vector z and label y (as one-hot), then generates images.
    """

    def __init__(
        self,
        latent_dim,
        hidden_dim_z,
        hidden_dim_y,
        combined_hidden_dim,
        img_shape,
        num_classes,
        use_dropout=True,
        dropout_prob=0.5,
    ):
        """
        Args:
            latent_dim: Dimension of latent vector z
            hidden_dim_z: Hidden layer size for z branch
            hidden_dim_y: Hidden layer size for y branch
            combined_hidden_dim: Hidden layer size after concatenation
            img_shape: Output image shape (C, H, W)
            num_classes: Number of classes (for one-hot)
            use_batch_norm: Whether to use batch norm
            use_dropout: Whether to use dropout
            dropout_prob: Dropout probability
        """
        super().__init__({"model": {"latent_dim": latent_dim}})
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.img_size = int(np.prod(img_shape))
        self.num_classes = num_classes

        self.z_mapper = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_z),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob) if use_dropout else nn.Identity(),
        )
        self.y_mapper = nn.Sequential(
            nn.Linear(num_classes, hidden_dim_y),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob) if use_dropout else nn.Identity(),
        )

        self.post_mapper = nn.Sequential(
            nn.Linear(hidden_dim_z + hidden_dim_y, combined_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob) if use_dropout else nn.Identity(),
            nn.Linear(combined_hidden_dim, self.img_size),
            nn.Tanh(),
        )

    def forward(self, z, y=None):
        """
        Args:
            z: Latent vector [batch_size, latent_dim]
            y: Class label (one-hot) [batch_size, num_classes]
        Returns:
            Generated images [batch_size, C, H, W]
        """
        if y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1):
            y = F.one_hot(y.view(-1), num_classes=self.num_classes)
        y = y.float()
        z_proj = self.z_mapper(z)
        y_proj = self.y_mapper(y)
        concat = torch.cat([z_proj, y_proj], dim=1)
        img_flat = self.post_mapper(concat)
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img

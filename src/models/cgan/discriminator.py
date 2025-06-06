"""
Conditional Discriminator for CGAN (MNIST, fully connected, one-hot label conditioning)
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.models.base.base_discriminator import BaseDiscriminator


class CGANDiscriminator(BaseDiscriminator):
    """
    Discriminator network for Conditional GAN.
    Receives image x and label y (as one-hot), outputs real/fake score.
    """

    def __init__(
        self,
        img_shape,
        hidden_dim_x,
        hidden_dim_y,
        combined_hidden_dim,
        num_classes,
        use_dropout=True,
        dropout_prob=0.5,
    ):
        """
        Args:
            img_shape: Shape of input image (C, H, W)
            hidden_dim_x: Hidden layer size for x branch
            hidden_dim_y: Hidden layer size for y branch
            combined_hidden_dim: Hidden layer size after concatenation
            num_classes: Number of classes (for one-hot)
            use_batch_norm: Whether to use batch norm
            use_dropout: Whether to use dropout
            dropout_prob: Dropout probability
        """
        super().__init__({"model": {}})
        self.img_shape = img_shape
        self.img_size = int(np.prod(img_shape))
        self.num_classes = num_classes

        self.x_mapper = nn.Sequential(
            nn.Linear(self.img_size, hidden_dim_x),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob) if use_dropout else nn.Identity(),
        )

        self.y_mapper = nn.Sequential(
            nn.Linear(num_classes, hidden_dim_y),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob) if use_dropout else nn.Identity(),
        )

        self.post_mapper = nn.Sequential(
            nn.Linear(hidden_dim_x + hidden_dim_y, combined_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob) if use_dropout else nn.Identity(),
            nn.Linear(combined_hidden_dim, 1),
        )

    def forward(self, img, y=None):
        """
        Args:
            img: Input images [batch_size, C, H, W]
            y: Class label (one-hot) [batch_size, num_classes]
        Returns:
            Real/fake logits [batch_size, 1]
        """
        if y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1):
            y = F.one_hot(y.view(-1), num_classes=self.num_classes)

        y = y.float()
        x_flat = img.view(img.size(0), -1)
        x_proj = self.x_mapper(x_flat)
        y_proj = self.y_mapper(y)
        concat = torch.cat([x_proj, y_proj], dim=1)
        validity = self.post_mapper(concat)
        return validity

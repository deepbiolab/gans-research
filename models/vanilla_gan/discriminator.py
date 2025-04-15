import torch
import torch.nn as nn
import numpy as np
from models.base.base_discriminator import BaseDiscriminator

class VanillaDiscriminator(BaseDiscriminator):
    """
    Discriminator network for the original GAN.
    Uses fully connected layers to classify images as real or fake.
    """
    def __init__(self, img_shape, hidden_dim):
        """
        Initialize the discriminator.
        
        Args:
            img_shape: Shape of the input image (channels, height, width)
            hidden_dim: Dimension of hidden layers
        """
        super().__init__({"model": {}})  # Empty config as we pass explicit parameters
        
        self.img_shape = img_shape
        self.hidden_dim = hidden_dim
        
        # Calculate total number of pixels in the image
        self.img_size = int(np.prod(img_shape))
        
        # Build the network
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(self.img_size, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second hidden layer
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third hidden layer
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer - no activation (using BCEWithLogitsLoss)
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, img):
        """
        Forward pass of the discriminator.
        
        Args:
            img: Input images of shape [batch_size, C, H, W]
            
        Returns:
            Classification scores (logits) for each image
        """
        # Flatten the images
        img_flat = img.view(img.size(0), -1)
        
        # Calculate discriminator score
        validity = self.model(img_flat)
        
        return validity
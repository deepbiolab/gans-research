import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDiscriminator(nn.Module, ABC):
    """
    Base class for all discriminator/critic implementations.
    This provides a common interface for different discriminator architectures.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input images of shape [batch_size, channels, height, width]
            
        Returns:
            Discrimination scores. For standard GANs, this is a single value per sample.
            For WGAN variants, this is a real-valued score.
        """
        pass
    
    def init_weights(self, module):
        """Initialize the weights of the network for better training."""
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
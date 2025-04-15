import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseGAN(nn.Module, ABC):
    """
    Base class for all GAN implementations.
    This provides a common interface for different GAN variants.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["model"]["latent_dim"]
        self.device = torch.device(config["experiment"]["device"])
        
        # These will be implemented by specific GAN variants
        self.generator = None
        self.discriminator = None
    
    @abstractmethod
    def build_generator(self):
        """Build the generator network."""
        pass
    
    @abstractmethod
    def build_discriminator(self):
        """Build the discriminator/critic network."""
        pass
    
    def generate_latent(self, batch_size):
        """Generate random latent vectors."""
        return torch.randn(batch_size, self.latent_dim).to(self.device)
    
    def generate_images(self, batch_size=None, latent_vectors=None):
        """Generate images from random latent vectors or provided ones."""
        if latent_vectors is None:
            if batch_size is None:
                raise ValueError("Either batch_size or latent_vectors must be provided")
            latent_vectors = self.generate_latent(batch_size)
            
        with torch.no_grad():
            images = self.generator(latent_vectors)
        
        return images
    
    @abstractmethod
    def generator_loss(self, *args, **kwargs):
        """Calculate the loss for the generator."""
        pass
    
    @abstractmethod
    def discriminator_loss(self, *args, **kwargs):
        """Calculate the loss for the discriminator/critic."""
        pass
    
    def save(self, path):
        """Save the model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
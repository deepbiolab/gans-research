"""
Conditional GAN module (CGAN) for MNIST.
"""
import torch
from src.models.base.base_gan import BaseGAN
from .generator import CGANGenerator
from .discriminator import CGANDiscriminator

class CGAN(BaseGAN):
    """
    Conditional GAN implementation for MNIST.
    """

    def __init__(self, config):
        super().__init__(config)

        self.latent_dim = config["model"]["latent_dim"]
        self.hidden_dim_z = config["model"]["hidden_dim_z"]
        self.hidden_dim_y = config["model"]["hidden_dim_y"]
        self.combined_hidden_dim = config["model"]["combined_hidden_dim"]
        self.img_shape = (
            config["data"]["channels"],
            config["data"]["image_size"],
            config["data"]["image_size"],
        )
        self.num_classes = config["data"]["num_classes"]

        # Build generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator.apply(self.generator.init_weights)
        self.discriminator.apply(self.discriminator.init_weights)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def build_generator(self):
        """Build the generator network."""
        return CGANGenerator(
            latent_dim=self.latent_dim,
            hidden_dim_z=self.hidden_dim_z,
            hidden_dim_y=self.hidden_dim_y,
            combined_hidden_dim=self.combined_hidden_dim,
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            use_dropout=self.config["model"].get("use_dropout", True),
            dropout_prob=self.config["model"].get("dropout_prob", 0.5),
        )

    def build_discriminator(self):
        """Build the discriminator network."""
        return CGANDiscriminator(
            img_shape=self.img_shape,
            hidden_dim_x=self.hidden_dim_z,
            hidden_dim_y=self.hidden_dim_y,
            combined_hidden_dim=self.combined_hidden_dim,
            num_classes=self.num_classes,
            use_dropout=self.config["model"].get("use_dropout", True),
            dropout_prob=self.config["model"].get("dropout_prob", 0.5),
        )
    
    def generate_images(self, batch_size=None, latent_vectors=None, labels=None):
        if latent_vectors is None:
            if batch_size is None:
                raise ValueError("Either batch_size or latent_vectors must be provided")
            latent_vectors = self.generate_latent(batch_size)
        else:
            batch_size = latent_vectors.shape[0]

        if labels is None:
            labels = torch.randint(
                0, self.generator.num_classes, (batch_size,), device=self.device
            )
        else:
            labels = labels.to(self.device)

        with torch.no_grad():
            images = self.generator(latent_vectors, labels)
        return images

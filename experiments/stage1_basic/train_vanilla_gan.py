"""
Training script for Vanilla GAN on MNIST dataset.
"""

import argparse

import yaml
import torch
from torch.utils.data import DataLoader

from src.models.vanilla_gan import VanillaGAN
from src.data.dataloader import create_dataloader
from src.training.trainer import GANTrainer
from src.utils.set_experiment import configure_experiment


class VanillaGANTrainer(GANTrainer):
    """
    Trainer specific to Vanilla GAN.
    Inherits from the base trainer and implements the train_step method.
    """

    def __init__(self, model: VanillaGAN, config: dict, dataloader: DataLoader):
        super().__init__(model, config, dataloader)

    def train_step(self, real_batch, iteration):
        """
        Single training step for Vanilla GAN.
        Trains the discriminator and generator.

        Args:
            real_batch: Batch of real images
            iteration: Current iteration number

        Returns:
            losses: Dictionary of losses
        """
        # Move data to device
        if isinstance(real_batch, (list, tuple)):
            real_imgs = real_batch[0].to(self.device)
        else:
            real_imgs = real_batch.to(self.device)

        batch_size = real_imgs.size(0)

        # -----------------
        #  Train Discriminator
        # -----------------
        self.d_optimizer.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, self.model.latent_dim).to(self.device)
        fake_imgs = self.model.generator(z)

        # Get discriminator outputs
        real_preds = self.model.discriminator(real_imgs)
        fake_preds = self.model.discriminator(fake_imgs.detach())

        # Calculate discriminator loss
        d_loss = self.model.discriminator_loss(real_preds, fake_preds)

        # Backpropagate and optimize
        d_loss.backward()
        self.d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        self.g_optimizer.zero_grad()

        # Generate new fake images (we do this again to get a fresh computation graph)
        z = torch.randn(batch_size, self.model.latent_dim).to(self.device)
        fake_imgs = self.model.generator(z)

        # Get discriminator predictions on generated images
        fake_preds = self.model.discriminator(fake_imgs)

        # Calculate generator loss
        g_loss = self.model.generator_loss(fake_preds)

        # Backpropagate and optimize
        g_loss.backward()
        self.g_optimizer.step()

        # Create dictionary of losses for logging
        losses = {"g_loss": g_loss.item(), "d_loss": d_loss.item()}  
        return losses


def main():
    """
    Main entry point for the Vanilla GAN training script.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Vanilla GAN on MNIST")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vanilla_gan.yaml",
        help="Path to the config file",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configure experiment environment
    config = configure_experiment(config, gpu_id=args.gpu)

    # Create dataloader
    dataloader = create_dataloader(config)

    # Create model
    model = VanillaGAN(config)

    # Create trainer and train
    trainer = VanillaGANTrainer(model, config, dataloader)
    trainer.train()


if __name__ == "__main__":
    main()

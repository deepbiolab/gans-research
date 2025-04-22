"""
Training script for Conditional GAN (CGAN) on MNIST or similar datasets.
"""

import argparse
import yaml

import torch
from src.models import CGAN
from src.losses import VanillaGANLoss
from src.training import GANTrainer
from src.data.dataloader import create_dataloader
from src.utils.set_experiment import configure_experiment


class CGANTrainer(GANTrainer):
    """
    Trainer for Conditional GAN (CGAN).
    """

    def train_step(self, real_batch, iteration):
        """
        Single training step for Conditional GAN (CGAN).
        Args:
            real_batch: (images, labels) tuple from DataLoader
            iteration: Current iteration number
        Returns:
            dict: Dictionary of losses
        """
        real_imgs, labels = real_batch
        real_imgs = real_imgs.to(self.device)
        labels = labels.to(self.device)
        batch_size = real_imgs.size(0)

        # -----------------
        #  Train Discriminator
        # -----------------
        self.d_optimizer.zero_grad()

        # Generate fake images with conditional labels
        z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_imgs = self.model.generator(z, labels)

        # Discriminator outputs for real and fake images (with labels)
        real_preds = self.model.discriminator(real_imgs, labels)
        fake_preds = self.model.discriminator(fake_imgs.detach(), labels)

        # Labels for loss
        real_targets = torch.ones_like(real_preds)
        fake_targets = torch.zeros_like(fake_preds)

        # Discriminator loss
        d_loss = self.criterion.discriminator_loss(
            real_preds, fake_preds, real_targets, fake_targets
        )

        d_loss.backward()
        self.d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        self.g_optimizer.zero_grad()

        # Generate new fake images for generator update
        z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_imgs = self.model.generator(z, labels)

        # Discriminator predictions on new fake images
        fake_preds = self.model.discriminator(fake_imgs, labels)
        fake_targets = torch.ones_like(fake_preds)  # Generator wants these to be real

        # Generator loss
        g_loss = self.criterion.generator_loss(fake_preds, fake_targets)

        g_loss.backward()
        self.g_optimizer.step()

        # Return losses for logging
        losses = {"g_loss": g_loss.item(), "d_loss": d_loss.item()}
        return losses


def main():
    """
    Main entry point for the Conditional GAN training script.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Conditional GAN on MNIST")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cgan.yaml",
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
    train_dataloader, valid_dataloader = create_dataloader(config)

    # Create model
    model = CGAN(config)

    # Create loss function
    loss_fn = VanillaGANLoss()

    # Create trainer and train
    trainer = CGANTrainer(
        model, config, train_dataloader, valid_dataloader, loss_fn=loss_fn
    )
    trainer.train()


if __name__ == "__main__":
    main()

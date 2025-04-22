"""
Training script for Wasserstein GAN with Gradient Penalty (WGAN-GP).

This script trains a WGAN-GP model on the specified dataset and
logs results to the configured output directory.
"""

import argparse
import yaml
import torch
from src.models import WGANGP
from src.losses import WGANGPLoss
from src.data.dataloader import create_dataloader
from src.training import GANTrainer
from src.utils.set_experiment import configure_experiment


class WGANGPTrainer(GANTrainer):
    """
    Trainer class for WGAN-GP.

    Extends the base GANTrainer with WGAN-GP specific training procedures.
    Implements multiple critic (discriminator) updates per generator update,
    and adds the gradient penalty term to the discriminator loss.
    """

    def train_step(self, real_batch, iteration):
        """
        Performs one training step for WGAN-GP.

        Args:
            real_batch: A batch of real images from the dataloader.
            iteration: Current training iteration (for logging/scheduling).

        Returns:
            dict: Losses for generator, discriminator, and gradient penalty.
        """
        # Prepare real images batch
        if isinstance(real_batch, (list, tuple)):
            real_imgs = real_batch[0].to(self.device)
        else:
            real_imgs = real_batch.to(self.device)
        batch_size = real_imgs.size(0)

        # Hyperparameters from config
        lambda_gp = self.config["training"].get("lambda_gp", 10.0)
        n_critic = self.config["training"].get("n_critic", 5)

        d_loss_total = 0.0
        gp_total = 0.0

        # Train the critic (discriminator) n_critic times
        for _ in range(n_critic):
            self.d_optimizer.zero_grad()
            # Sample random noise for generator
            z = torch.randn(batch_size, self.model.latent_dim, 1, 1, device=self.device)
            fake_imgs = self.model.generator(z)
            # Critic predictions
            real_preds = self.model.discriminator(real_imgs)
            fake_preds = self.model.discriminator(fake_imgs.detach())
            # Wasserstein loss (reuse WGANLoss)
            wasserstein_d_loss = self.criterion.discriminator_loss(
                real_preds, fake_preds
            )
            # Gradient penalty
            gp = self.criterion.gradient_penalty(
                self.model.discriminator, real_imgs, fake_imgs, lambda_gp
            )
            # Total discriminator loss
            d_loss = wasserstein_d_loss + gp
            d_loss.backward()
            self.d_optimizer.step()
            d_loss_total += d_loss.item()
            gp_total += gp.item()

        d_loss_avg = d_loss_total / n_critic
        gp_avg = gp_total / n_critic

        # Generator update (1 step)
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.model.latent_dim, 1, 1, device=self.device)
        fake_imgs = self.model.generator(z)
        fake_preds = self.model.discriminator(fake_imgs)
        g_loss = self.criterion.generator_loss(fake_preds)
        g_loss.backward()
        self.g_optimizer.step()

        # Return losses for logging
        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss_avg,
            "gp_loss": gp_avg,
        }


def main():
    """
    Main entry point for the WGANGP training script.
    """
    parser = argparse.ArgumentParser(description="Train WGAN-GP on a dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wgan_gp.yaml",
        help="Path to the config file",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configure experiment environment
    config = configure_experiment(config, gpu_id=args.gpu)

    # Create dataloaders
    train_dataloader, valid_dataloader = create_dataloader(config)

    # Create model
    model = WGANGP(config)

    # Create loss function
    loss_fn = WGANGPLoss()

    # Create WGAN-GP Trainer and start training
    trainer = WGANGPTrainer(
        model, config, train_dataloader, valid_dataloader, loss_fn=loss_fn
    )
    trainer.train()


if __name__ == "__main__":
    main()

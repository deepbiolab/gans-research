"""
Training script for DCGAN on CeleBA dataset with training tricks.
"""

import argparse
import yaml
import torch
from src.models import DCGAN
from src.losses import VanillaGANLoss
from src.training import GANTrainer
from src.data.dataloader import create_dataloader
from src.data.data_utils import (
    augment,
    get_positive_labels,
    get_negative_labels,
)
from src.utils.set_experiment import configure_experiment

try:
    from ema_pytorch import EMA
except ImportError:
    EMA = None


class DCGANTrainer(GANTrainer):
    """
    DCGAN Trainer with support for training tricks (Augment, EMA, label smoothing, random label flip).
    """

    def __init__(
        self,
        model,
        config,
        train_dataloader,
        valid_dataloader,
        loss_fn,
    ):
        super().__init__(model, config, train_dataloader, valid_dataloader, loss_fn)

        # --- Training tricks configuration ---
        tricks = config

        # Augment images
        self.use_augment = tricks.get("augment", {}).get("enable", False)
        self.augment_policy = tricks.get("augment", {}).get("policy", "")

        # Label smoothing
        self.use_label_smoothing = tricks.get("label_smoothing", {}).get(
            "enable", False
        )

        # Random label flip
        self.use_random_label_flip = tricks.get("random_label_flip", {}).get(
            "enable", False
        )
        self.random_flip_prob = tricks.get("random_label_flip", {}).get("prob", 0.0)

        # EMA (Exponential Moving Average)
        self.use_ema = tricks.get("ema", {}).get("enable", False)
        self.ema = None
        if self.use_ema and EMA is not None:
            self.ema = EMA(
                self.model.generator,
                beta=tricks["ema"].get("beta", 0.995),
                update_after_step=tricks["ema"].get("update_after_step", 100),
                update_every=tricks["ema"].get("update_every", 1),
            )
            self.logger.info("EMA enabled for generator.")
        elif self.use_ema:
            self.logger.warning("EMA requested but ema_pytorch is not installed.")

    def train_step(self, real_batch, iteration):
        """
        Single training step with tricks applied.
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

        # Augment on real images if enabled
        if self.use_augment:
            real_imgs_aug = augment(real_imgs, policy=self.augment_policy)
        else:
            real_imgs_aug = real_imgs

        # Generate fake images
        z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_imgs = self.model.generator(z)

        # Augment on fake images if enabled
        if self.use_augment:
            fake_imgs_aug = augment(fake_imgs, policy=self.augment_policy)
        else:
            fake_imgs_aug = fake_imgs

        # --- Label tricks ---
        # Positive labels for real images
        real_labels = get_positive_labels(
            batch_size,
            self.device,
            smoothing=self.use_label_smoothing,
            random_flip=self.random_flip_prob if self.use_random_label_flip else 0.0,
        )
        # Negative labels for fake images
        fake_labels = get_negative_labels(batch_size, self.device)

        # Discriminator predictions
        real_preds = self.model.discriminator(real_imgs_aug).view(-1)
        fake_preds = self.model.discriminator(fake_imgs_aug.detach()).view(-1)

        # Discriminator loss with label tricks
        d_loss = self.criterion.discriminator_loss(real_preds, fake_preds, real_labels, fake_labels)

        d_loss.backward()
        self.d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        self.g_optimizer.zero_grad()

        # Generate new fake images for generator update
        z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_imgs = self.model.generator(z)

        # Augment on fake images if enabled
        if self.use_augment:
            fake_imgs_aug = augment(fake_imgs, policy=self.augment_policy)
        else:
            fake_imgs_aug = fake_imgs

        # Generator tries to fool discriminator (labels are all real/1)
        fake_labels = torch.full((batch_size,), 1.0, device=self.device)
        fake_preds = self.model.discriminator(fake_imgs_aug).view(-1)
        g_loss = self.criterion.generator_loss(fake_preds, fake_labels)

        g_loss.backward()
        self.g_optimizer.step()

        # EMA update
        if self.ema is not None:
            self.ema.update()

        # Return loss dictionary for logging
        losses = {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
        }
        return losses

    def generate_samples(self, sampling_num: int = 16) -> torch.Tensor:
        """
        Generate and return sample images using (optionally) EMA weights.
        """
        self.model.eval()
        # Use EMA weights if enabled and available
        generator = (
            self.ema.ema_model if (self.ema is not None) else self.model.generator
        )
        with torch.no_grad():
            z = torch.randn(sampling_num, self.model.latent_dim, device=self.device)
            samples = generator(z)
        self.model.train()
        return samples


def main():
    """
    Main entry point for the DCGAN training script.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train DCGAN on MNIST")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dcgan.yaml",
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
    model = DCGAN(config)

    # Create loss function
    loss_fn = VanillaGANLoss()

    # Create DCGANTrainer and train
    trainer = DCGANTrainer(
        model, config, train_dataloader, valid_dataloader, loss_fn=loss_fn
    )
    trainer.train()


if __name__ == "__main__":
    main()

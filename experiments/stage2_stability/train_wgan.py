"""
Training script for WGAN on a dataset with basic training tricks.
"""

import argparse
import yaml
import torch
from src.models import WGAN
from src.losses import WGANLoss
from src.data.dataloader import create_dataloader
from src.data.data_utils import augment
from src.training import GANTrainer
from src.utils.set_experiment import configure_experiment

try:
    from ema_pytorch import EMA
except ImportError:
    EMA = None


class WGANTrainer(GANTrainer):
    """
    Trainer for Wasserstein GAN (WGAN) supporting n_critic, weight clipping, EMA, and data augmentation.

    Args:
        model: The GAN model (generator + discriminator/critic).
        config: Dictionary of experiment and training configuration.
        train_dataloader: PyTorch DataLoader for training data.
        valid_dataloader: PyTorch DataLoader for validation data.
        loss_fn: WGAN loss function.
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

        # n_critic: number of critic updates per generator update
        self.n_critic = config["training"].get("n_critic", 5)

        # Weight clipping
        self.use_weight_clipping = config["training"].get("use_weight_clipping", True)
        self.clip_value = config["training"].get("clip_value", [-0.01, 0.01])

        # EMA (Exponential Moving Average) for generator
        self.use_ema = config.get("ema", {}).get("enable", False)
        self.ema = None
        if self.use_ema and EMA is not None:
            self.ema = EMA(
                self.model.generator,
                beta=config["ema"].get("beta", 0.995),
                update_after_step=config["ema"].get("update_after_step", 100),
                update_every=config["ema"].get("update_every", 1),
            )
            self.logger.info("EMA enabled for generator.")
        elif self.use_ema:
            self.logger.warning("EMA requested but ema_pytorch is not installed.")

        # Data augmentation (optional)
        self.use_augment = config.get("augment", {}).get("enable", False)
        self.augment_policy = config.get("augment", {}).get("policy", "")

    def train_step(self, real_batch, iteration):
        """
        Performs a single training step for WGAN.

        Args:
            real_batch: A batch of real images.
            iteration: Current iteration number.

        Returns:
            dict: Dictionary containing generator and critic losses.
        """
        if isinstance(real_batch, (list, tuple)):
            real_imgs = real_batch[0].to(self.device)
        else:
            real_imgs = real_batch.to(self.device)
        batch_size = real_imgs.size(0)

        # -----------------
        #  Train Critic (Discriminator)
        # -----------------
        for _ in range(self.n_critic):
            self.d_optimizer.zero_grad()

            # Optional data augmentation
            if self.use_augment:
                real_imgs_aug = augment(real_imgs, policy=self.augment_policy)
            else:
                real_imgs_aug = real_imgs

            # Generate fake images
            z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
            fake_imgs = self.model.generator(z).detach()
            if self.use_augment:
                fake_imgs_aug = augment(fake_imgs, policy=self.augment_policy)
            else:
                fake_imgs_aug = fake_imgs

            # Critic predictions
            real_preds = self.model.discriminator(real_imgs_aug)
            fake_preds = self.model.discriminator(fake_imgs_aug)

            # WGAN loss (no labels)
            d_loss = self.criterion.discriminator_loss(real_preds, fake_preds)
            d_loss.backward()
            self.d_optimizer.step()

            # Weight clipping (WGAN original)
            if self.use_weight_clipping:
                for p in self.model.discriminator.parameters():
                    p.data.clamp_(self.clip_value[0], self.clip_value[1])

        # -----------------
        #  Train Generator
        # -----------------
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_imgs = self.model.generator(z)
        if self.use_augment:
            fake_imgs_aug = augment(fake_imgs, policy=self.augment_policy)
        else:
            fake_imgs_aug = fake_imgs

        fake_preds = self.model.discriminator(fake_imgs_aug)
        g_loss = self.criterion.generator_loss(fake_preds)
        g_loss.backward()
        self.g_optimizer.step()

        # EMA update (if enabled)
        if self.ema is not None:
            self.ema.update()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
        }

    def generate_samples(self, sampling_num: int = 16) -> torch.Tensor:
        """
        Generate and return sample images using (optionally) EMA weights.

        Args:
            sampling_num: Number of samples to generate.

        Returns:
            torch.Tensor: Generated sample images.
        """
        self.model.eval()
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
    Main entry point for the WGAN training script.
    """
    parser = argparse.ArgumentParser(description="Train WGAN on a dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wgan.yaml",
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
    model = WGAN(config)

    # Create loss function
    loss_fn = WGANLoss()

    # Create WGANTrainer and train
    trainer = WGANTrainer(
        model, config, train_dataloader, valid_dataloader, loss_fn=loss_fn
    )
    trainer.train()


if __name__ == "__main__":
    main()

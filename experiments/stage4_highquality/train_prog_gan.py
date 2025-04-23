"""
Training script for ProGAN with WGAN-GP, progressive growing, and fade-in.

This script implements the Progressive Growing of GANs methodology (Karras et al., 2018)
with Wasserstein GAN with Gradient Penalty (WGAN-GP) loss function. The training
progressively increases the resolution of generated images, starting from 4x4 and
gradually growing to higher resolutions.
"""

import argparse
import time
import yaml
import torch
from tqdm import tqdm
from src.models import ProgGAN
from src.losses import ProgGANGPLoss
from src.training import GANTrainer
from src.data.dataloader import create_dataloader
from src.utils.set_experiment import configure_experiment
from src.utils.visualization import create_animation


class ProGANGPTrainer(GANTrainer):
    """
    ProGAN Trainer: Supports progressive growing (fade-in), WGAN-GP loss, and drift penalty.

    This trainer extends the base GANTrainer to implement progressive growing of GANs,
    where the resolution of generated images increases gradually during training.
    It also implements the WGAN-GP loss function with an additional drift penalty term.

    Args:
        model: The ProgGAN model
        config: Configuration dictionary
        train_dataloader: Training data loader
        valid_dataloader: Validation data loader
        loss_fn: Loss function (ProgGANGPLoss)
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

        # Drift penalty coefficient to prevent discriminator values from drifting too far from zero
        self.epsilon_drift = config.get("training", {}).get("epsilon_drift", 0.001)

        # Number of discriminator updates per generator update
        self.n_critic = config.get("training", {}).get("n_critic", 1)

        # Progressive growing configuration
        self.progressive_cfg = config.get("progressive", {})
        # List of resolutions to progress through (e.g., [4, 8, 16, 32, 64, 128])
        self.resolutions = self.progressive_cfg.get(
            "resolutions", [4, 8, 16, 32, 64, 128, 256, 512]
        )
        # Number of images to train at each resolution stage
        self.images_per_stage = self.progressive_cfg.get("images_per_stage", 100_000)
        # Number of images over which to fade in new layers
        self.fadein_kimgs = self.progressive_cfg.get("fadein_kimgs", 100_000)
        # Current resolution stage index
        self.stage = 0
        # Current resolution (e.g., 4, 8, 16, etc.)
        self.current_res = self.resolutions[self.stage]
        # Fade-in coefficient (0.0 to 1.0)
        self.alpha = 0.0
        # Counter for number of images processed during training
        self.images_seen = 0
        # Number of images for the fade-in phase
        self.fadein_images = self.fadein_kimgs
        # Total images to process in each resolution stage
        self.total_images_in_stage = self.images_per_stage

        # Create dataloaders with the initial resolution
        self.train_dataloader, self.valid_dataloader = create_dataloader(
            self.config, override_image_size=self.current_res
        )

    def update_progressive(self, batch_size):
        """
        Update resolution stage and alpha value to implement progressive growing and fade-in.

        This method tracks the number of images processed and updates the alpha value
        for smooth transition between resolution stages. When enough images have been
        processed at the current stage, it moves to the next resolution.

        Args:
            batch_size: Number of images in the current batch

        Returns:
            bool: True if resolution changed and dataloaders need to be rebuilt
        """
        # Increment the counter of processed images
        self.images_seen += batch_size

        # Update alpha during fade-in phase
        if self.images_seen < self.fadein_images:
            self.alpha = self.images_seen / float(self.fadein_images)
        else:
            # After fade-in is complete, alpha stays at 1.0
            self.alpha = 1.0

        # Check if it's time to move to the next resolution stage
        if self.images_seen >= self.total_images_in_stage:
            if self.stage + 1 < len(self.resolutions):
                # Move to next resolution stage
                self.stage += 1
                self.current_res = self.resolutions[self.stage]
                # Reset counters for the new stage
                self.images_seen = 0
                self.alpha = 0.0
                self.logger.info(
                    f"Stage up! Now at resolution {self.current_res}x{self.current_res}"
                )
                return True  # Signal that dataloaders need to be rebuilt
            else:
                # We've reached the final resolution
                self.current_res = self.resolutions[-1]
                self.alpha = 1.0
        return False

    def reload_dataloader(self):
        """
        Recreate dataloaders to match the current resolution.

        This method is called when the resolution changes to ensure that
        the real images match the current generator resolution.
        """
        self.train_dataloader, self.valid_dataloader = create_dataloader(
            self.config, override_image_size=self.current_res
        )

    def train_step(self, real_batch, iteration):
        """
        Execute a single training step for both generator and discriminator.

        This method implements the WGAN-GP training procedure with n_critic
        discriminator updates per generator update. It also applies the
        progressive growing methodology by using the current resolution and alpha.

        Args:
            real_batch: Batch of real images
            iteration: Current training iteration

        Returns:
            dict: Dictionary of loss values and training metrics
        """
        # 1. Get real images and adjust for current resolution
        if isinstance(real_batch, (list, tuple)):
            real_imgs = real_batch[0].to(self.device)
        else:
            real_imgs = real_batch.to(self.device)

        batch_size = real_imgs.size(0)

        # 2. Train discriminator for n_critic iterations
        d_loss_total = 0
        gp_total = 0.0

        for _ in range(self.n_critic):
            # Zero gradients for discriminator
            self.d_optimizer.zero_grad()

            # Generate random latent vectors and fake images
            z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
            fake_imgs = self.model.generator(
                z, current_res=self.current_res, alpha=self.alpha
            )

            # Verify that real and fake images have the same shape
            assert (
                real_imgs.shape == fake_imgs.shape
            ), f"real: {real_imgs.shape}, fake: {fake_imgs.shape}"

            # Get discriminator predictions for real and fake images
            real_preds = self.model.discriminator(
                real_imgs, current_res=self.current_res, alpha=self.alpha
            )
            fake_preds = self.model.discriminator(
                fake_imgs.detach(), current_res=self.current_res, alpha=self.alpha
            )

            # Calculate WGAN-GP loss components
            wasserstein_loss = self.criterion.discriminator_loss(real_preds, fake_preds)
            gp = self.criterion.gradient_penalty(
                self.model.discriminator,
                real_imgs,
                fake_imgs.detach(),
                lambda_gp=self.config["training"].get("lambda_gp", 10.0),
                current_res=self.current_res,
                alpha=self.alpha,
            )

            # Add drift penalty to prevent discriminator values from drifting too far
            drift = self.epsilon_drift * (real_preds**2).mean()

            # Combine all loss components
            d_loss = wasserstein_loss + gp + drift

            # Backpropagate and update discriminator
            d_loss.backward()
            self.d_optimizer.step()

            # Accumulate loss values for logging
            d_loss_total += d_loss.item()
            gp_total += gp.item()

        # Calculate average discriminator loss over n_critic iterations
        d_loss_avg = d_loss_total / self.n_critic
        gp_avg = gp_total / self.n_critic

        # 3. Train generator
        self.g_optimizer.zero_grad()

        # Generate new fake images for generator update
        z = torch.randn(batch_size, self.model.latent_dim, device=self.device)
        fake_imgs = self.model.generator(
            z, current_res=self.current_res, alpha=self.alpha
        )

        # Get discriminator predictions for the fake images
        fake_preds = self.model.discriminator(
            fake_imgs, current_res=self.current_res, alpha=self.alpha
        )

        # Calculate generator loss
        g_loss = self.criterion.generator_loss(fake_preds)

        # Backpropagate and update generator
        g_loss.backward()
        self.g_optimizer.step()

        # 4. Return loss values and training metrics
        losses = {
            "g_loss": g_loss.item(),
            "d_loss": d_loss_avg,
            "gp_loss": gp_avg,
            "alpha": float(self.alpha),
            "current_res": int(self.current_res),
        }
        return losses

    def generate_samples(self, sampling_num: int = 16) -> torch.Tensor:
        """
        Generate sample images using the current generator.

        Args:
            sampling_num: Number of images to generate

        Returns:
            torch.Tensor: Batch of generated images
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.generate_images(
                batch_size=sampling_num,
                current_res=self.current_res,
                alpha=self.alpha,
            )
        self.model.train()
        return samples

    def train(self):
        """
        Main training loop for Progressive Growing GAN.

        This method implements the progressive growing training procedure,
        automatically rebuilding dataloaders when the resolution changes
        to ensure that real and fake images have matching resolutions.
        """
        self.logger.info("Using device: %s", self.device)
        self.logger.info(
            "Starting ProGAN progressive training for %d epochs", self.num_epochs
        )

        start_time = time.time()
        epoch = 0
        while epoch < self.num_epochs:
            self.logger.info(
                "Starting epoch %d/%d at resolution %dx%d",
                epoch + 1,
                self.num_epochs,
                self.current_res,
                self.current_res,
            )
            for real_batch in tqdm(self.train_dataloader):
                # Move data to device
                if isinstance(real_batch, (list, tuple)):
                    real_batch = [item.to(self.device) for item in real_batch]
                else:
                    real_batch = real_batch.to(self.device)

                # Execute training step
                losses = self.train_step(real_batch, self.iteration)

                # Log progress at specified intervals
                if self.iteration % self.log_interval == 0:
                    self.log_progress(losses, epoch, self.iteration, real_batch)

                # Save model checkpoint at specified intervals
                if self.iteration % self.save_interval == 0:
                    self.save_checkpoint(epoch, self.iteration)

                # Check if resolution needs to be updated
                reload_needed = self.update_progressive(
                    real_batch[0].shape[0]
                    if isinstance(real_batch, (list, tuple))
                    else real_batch.shape[0]
                )
                self.iteration += 1

                # If resolution changed, rebuild dataloaders and break the inner loop
                if reload_needed:
                    self.reload_dataloader()
                    break  # Exit the dataloader loop to restart with new resolution

            # Update learning rate if scheduler is used
            if self.g_scheduler is not None:
                self.g_scheduler.step()
            if self.d_scheduler is not None:
                self.d_scheduler.step()

            # Save checkpoint at the end of each epoch
            self.save_checkpoint(epoch, self.iteration, is_epoch_end=True)

            # Evaluate FID score at specified intervals
            if self.eval_fid and epoch % self.eval_epoch_interval == 0:
                self.evaluate_fid(self.iteration)

            epoch += 1

        # Calculate and log total training time
        total_time = time.time() - start_time
        self.logger.info("Training completed in %.2f hours", total_time / 3600)

        # Save final model
        self.save_checkpoint(self.num_epochs, self.iteration, is_final=True)

        # Create animation from saved sample images
        gif_path = create_animation(
            experiment_dir=self.output_dir,
            samples_subdir="samples",
            pattern="progress_*.png",
            include_iteration_text=True,
        )
        self.logger.info("Created samples animation: %s", gif_path)

        # Log animation to tensorboard if available
        if self.writer is not None:
            try:
                self.writer.add_artifact(
                    artifact_name="training_animation",
                    artifact_type="animation",
                    file_path=gif_path,
                    aliases=["final"],
                )
                self.writer.close()
            except AttributeError:
                self.logger.warning("Could not log artifact to tensorboard")


def main():
    """
    Main entry point for ProGAN training script.

    This function parses command-line arguments, loads the configuration,
    sets up the experiment environment, creates the model and trainer,
    and starts the training process.
    """
    parser = argparse.ArgumentParser(
        description="Train ProGAN with WGAN-GP and progressive growing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/prog_gan.yaml",
        help="Path to the config file",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configure experiment environment (output directories, logging, etc.)
    config = configure_experiment(config, gpu_id=args.gpu)

    # Create model from configuration
    model = ProgGAN(config)

    # Create loss function for ProGAN with gradient penalty
    loss_fn = ProgGANGPLoss()

    # Create trainer (dataloaders are created inside the trainer)
    trainer = ProGANGPTrainer(model, config, None, None, loss_fn=loss_fn)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

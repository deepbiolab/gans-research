"""
Base trainer for GAN models.
"""

import os
import time
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from src.models.base.base_gan import BaseGAN
from src.utils.set_experiment import setup_logger, setup_summary
from src.utils.visualization import make_grid, save_grid, create_animation


class GANTrainer(ABC):
    """
    Base trainer for GAN models.
    This provides a common interface for training different GAN variants.
    """

    def __init__(self, model: BaseGAN, config: dict, dataloader: DataLoader) -> None:
        self.config = config
        self.device = torch.device(config["experiment"]["device"])

        # Setup model and dataloader
        self.model = model
        self.dataloader = dataloader

        # Setup optimizers
        self.setup_optimizers()

        # Setup training parameters
        self.iteration = 0
        self.num_epochs = config["experiment"].get("num_epochs", 10)
        self.log_interval = config["experiment"].get("log_interval", 100)
        self.save_interval = config["experiment"].get("save_interval", 1000)
        self.sample_interval = config["experiment"].get("sample_interval", 500)
        self.num_samples = config["experiment"].get("num_samples", 16)

        # Setup output directory
        self.setup_directory()

        # Setup tensorboard or wandb
        self.writer = setup_summary(config, self.output_dir)

        # Setup logging
        self.logger = setup_logger(self.output_dir)

    def setup_directory(self):
        """
        Setup directories for output and checkpoints.
        """
        self.output_dir = self.config["experiment"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)

    def setup_optimizers(self):
        """
        Setup optimizers for generator and discriminator.
        Can be overridden by specific trainers.
        """
        optim_name = self.config["training"].get("optimizer", "adam").lower()
        lr_g = self.config["training"]["lr_g"]
        lr_d = self.config["training"]["lr_d"]
        beta1 = self.config["training"].get("beta1", 0.5)
        beta2 = self.config["training"].get("beta2", 0.999)

        if optim_name == "adam":
            self.g_optimizer = torch.optim.Adam(
                self.model.generator.parameters(), lr=lr_g, betas=(beta1, beta2)
            )
            self.d_optimizer = torch.optim.Adam(
                self.model.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2)
            )
        elif optim_name == "rmsprop":
            self.g_optimizer = torch.optim.RMSprop(
                self.model.generator.parameters(), lr=lr_g
            )
            self.d_optimizer = torch.optim.RMSprop(
                self.model.discriminator.parameters(), lr=lr_d
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")

        # Setup learning rate schedulers if needed
        if self.config["training"].get("use_scheduler", False):
            gamma = self.config["training"].get("scheduler_gamma", 0.99)
            self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.g_optimizer, gamma=gamma
            )
            self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.d_optimizer, gamma=gamma
            )
        else:
            self.g_scheduler = None
            self.d_scheduler = None

    @abstractmethod
    def train_step(self, real_batch, iteration):
        """
        Single training step.
        Must be implemented by specific GAN trainers.

        Args:
            real_batch: Batch of real data
            iteration: Current iteration number

        Returns:
            dict: Dictionary of losses
        """

    def train(self):
        """
        Main training loop.
        """
        self.logger.info("Using device: %s", self.device)
        self.logger.info("Starting training for %d epochs", self.num_epochs)

        start_time = time.time()
        for epoch in range(self.num_epochs):
            self.logger.info("Starting epoch %d/%d", epoch + 1, self.num_epochs)

            for real_batch in tqdm(self.dataloader):
                # Move data to device
                if isinstance(real_batch, (list, tuple)):
                    real_batch = [item.to(self.device) for item in real_batch]
                else:
                    real_batch = real_batch.to(self.device)

                # Train step
                losses = self.train_step(real_batch, self.iteration)

                # Logging
                if self.iteration % self.log_interval == 0:
                    self.log_progress(losses, epoch, self.iteration, real_batch)

                # Save model
                if self.iteration % self.save_interval == 0:
                    self.save_checkpoint(epoch, self.iteration)

                self.iteration += 1

            # Update learning rate if scheduler is used
            if self.g_scheduler is not None:
                self.g_scheduler.step()
            if self.d_scheduler is not None:
                self.d_scheduler.step()

            # Save at the end of each epoch
            self.save_checkpoint(epoch, self.iteration, is_epoch_end=True)

        total_time = time.time() - start_time
        self.logger.info("Training completed in %.2f hours", total_time / 3600)

        # Save final model
        self.save_checkpoint(self.num_epochs, self.iteration, is_final=True)

        # Create animation
        gif_path = create_animation(
            experiment_dir=self.output_dir,
            samples_subdir="samples",
            pattern="progress_*.png",
            include_iteration_text=True,
        )
        self.logger.info("Created samples animation: %s", gif_path)

    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """
        Generate and save sample images.
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.generate_images(num_samples)

        self.model.train()
        return samples

    def log_progress(
        self,
        losses: dict,
        epoch: int,
        iteration: int,
        real_batch: torch.Tensor,
        nrow: int = 8,
        image_name: str = "progress",
    ):
        """
        Log training progress.
        """
        # Log to console and file
        log_str = f"[Epoch {epoch+1}/{self.num_epochs}] [Iter {iteration}]"
        for name, value in losses.items():
            log_str += f" {name}: {value:.4f}"
        self.logger.info(log_str)

        # Generate samples
        samples = self.generate_samples(self.num_samples)
        grid = make_grid(samples, nrow=nrow)

        # Save grid of generated images
        filepath = os.path.join(self.samples_dir, f"{image_name}_{iteration:06d}.png")
        save_grid(grid, filepath)
        self.logger.info("Generated samples at iteration %d", iteration)

        # Log to tensorboard or wandb
        if self.writer is not None:
            # Log losses to tensorboard/wandb
            for name, value in losses.items():
                self.writer.add_scalar(f"loss/{name}", value, iteration)

            # Extract images from batch for comparison
            if isinstance(real_batch, (list, tuple)):
                images = real_batch[0].detach().cpu()
            else:
                images = real_batch.detach().cpu()

            num_samples = samples.size(0)
            if images.size(0) > num_samples:
                images = images[:num_samples]

            # Log real images to tensorboard/wandb
            self.writer.add_images(
                "samples/real", images, iteration, dataformats="NCHW"
            )

            # Log sample images to tensorboard/wandb
            self.writer.add_images(
                "samples/generated", samples, iteration, dataformats="NCHW"
            )

    def save_checkpoint(
        self,
        epoch: int,
        iteration: int,
        is_epoch_end: bool = False,
        is_final: bool = False,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current training epoch
            iteration: Current training iteration
            is_epoch_end: Whether this is an end-of-epoch checkpoint
            is_final: Whether this is the final model checkpoint

        Returns:
            None
        """
        filename = (
            "final_model.pth" if is_final else f"epoch_{epoch}_iter_{iteration}.pth"
        )
        if is_epoch_end and not is_final:
            filename = f"epoch_{epoch}_end.pth"

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "iteration": iteration,
                "generator_state_dict": self.model.generator.state_dict(),
                "discriminator_state_dict": self.model.discriminator.state_dict(),
                "g_optimizer_state_dict": self.g_optimizer.state_dict(),
                "d_optimizer_state_dict": self.d_optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        self.logger.info("Model saved to %s", path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
        map_location: str = None,
    ) -> dict:
        """
        Load model and optimizer state from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            load_optimizer: Whether to load optimizer states
            strict: Whether to strictly enforce that the keys in state_dict match the keys in model
            map_location: Optional device mapping when loading model on a different device

        Returns:
            dict: Loaded checkpoint data for further processing
        """
        self.logger.info("Loading checkpoint from %s", checkpoint_path)

        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load generator weights
        self.model.generator.load_state_dict(
            checkpoint["generator_state_dict"], strict=strict
        )

        # Load discriminator weights
        self.model.discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"], strict=strict
        )

        # Restore training state if needed
        self.iteration = checkpoint.get("iteration", 0)
        epoch = checkpoint.get("epoch", 0)

        # Load optimizer states if requested
        if load_optimizer and "g_optimizer_state_dict" in checkpoint:
            self.g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
            self.d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
            self.logger.info("Optimizer states loaded")

        self.logger.info(
            "Checkpoint loaded successfully, resuming from epoch %d, iteration %d",
            epoch,
            self.iteration,
        )

        return checkpoint

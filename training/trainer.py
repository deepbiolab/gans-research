import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

class GANTrainer:
    """
    Base trainer for GAN models.
    This provides a common interface for training different GAN variants.
    """
    def __init__(self, model, config, dataloader):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.device = torch.device(config["experiment"]["device"])
        
        # Training parameters
        self.num_epochs = config["experiment"].get("num_epochs", 100)
        self.log_interval = config["experiment"].get("log_interval", 100)
        self.save_interval = config["experiment"].get("save_interval", 1000)
        
        # Setup output directory
        self.output_dir = config["experiment"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
        
        # Setup tensorboard
        if config["logging"].get("use_tensorboard", True):
            self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        else:
            self.writer = None
        
        # Setup optimizers
        self.setup_optimizers()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
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
                self.model.generator.parameters(),
                lr=lr_g, betas=(beta1, beta2)
            )
            self.d_optimizer = torch.optim.Adam(
                self.model.discriminator.parameters(),
                lr=lr_d, betas=(beta1, beta2)
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
    
    def train_step(self, real_batch, iteration):
        """
        Single training step.
        To be implemented by specific GAN trainers.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def train(self):
        """
        Main training loop.
        """
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.iteration = 0
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            
            for real_batch in tqdm(self.dataloader):
                # Move data to device
                if isinstance(real_batch, list) or isinstance(real_batch, tuple):
                    real_batch = [item.to(self.device) for item in real_batch]
                else:
                    real_batch = real_batch.to(self.device)
                
                # Train step
                losses = self.train_step(real_batch, self.iteration)
                
                # Logging
                if self.iteration % self.log_interval == 0:
                    self.log_progress(losses, epoch, self.iteration)
                
                # Save model
                if self.iteration % self.save_interval == 0:
                    self.save_checkpoint(epoch, self.iteration)
                    self.generate_samples(self.iteration)
                
                self.iteration += 1
            
            # Update learning rate if scheduler is used
            if self.g_scheduler is not None:
                self.g_scheduler.step()
            if self.d_scheduler is not None:
                self.d_scheduler.step()
            
            # Save at the end of each epoch
            self.save_checkpoint(epoch, self.iteration, is_epoch_end=True)
            self.generate_samples(self.iteration, is_epoch_end=True)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Save final model
        self.save_checkpoint(self.num_epochs, self.iteration, is_final=True)
    
    def log_progress(self, losses, epoch, iteration):
        """
        Log training progress.
        """
        # Log to console and file
        log_str = f"[Epoch {epoch+1}/{self.num_epochs}] [Iter {iteration}]"
        for name, value in losses.items():
            log_str += f" {name}: {value:.4f}"
        self.logger.info(log_str)
        
        # Log to tensorboard
        if self.writer is not None:
            for name, value in losses.items():
                self.writer.add_scalar(f"loss/{name}", value, iteration)
    
    def save_checkpoint(self, epoch, iteration, is_epoch_end=False, is_final=False):
        """
        Save model checkpoint.
        """
        filename = "final_model.pth" if is_final else f"epoch_{epoch}_iter_{iteration}.pth"
        if is_epoch_end and not is_final:
            filename = f"epoch_{epoch}_end.pth"
            
        path = os.path.join(self.output_dir, "checkpoints", filename)
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def generate_samples(self, iteration, is_epoch_end=False):
        """
        Generate and save sample images.
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.generate_images(16)  # Generate 16 samples
        
        # Save samples code would be implemented
        # This requires visualization utilities
        self.logger.info(f"Generated samples at iteration {iteration}")
        self.model.train()
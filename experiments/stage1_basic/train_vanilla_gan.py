#!/usr/bin/env python
import os
import sys
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import our modules
from models.vanilla_gan import VanillaGAN
from data.dataloader import create_dataloader
from training.trainer import GANTrainer
from evaluation.visualization.grid_visualization import visualize_training_progress

class VanillaGANTrainer(GANTrainer):
    """
    Trainer specific to Vanilla GAN.
    Inherits from the base trainer and implements the train_step method.
    """
    def __init__(self, model, config, dataloader):
        super().__init__(model, config, dataloader)
        
        # Create fixed noise for visualization
        self.fixed_noise = torch.randn(16, self.model.latent_dim).to(self.device)
    
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
        if isinstance(real_batch, list) or isinstance(real_batch, tuple):
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
        losses = {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item()
        }
        
        # Generate and save images periodically
        if iteration % 500 == 0:
            visualize_training_progress(
                self.model.generator,
                self.device,
                self.fixed_noise,
                self.output_dir,
                iteration
            )
        
        return losses


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Vanilla GAN on MNIST")
    parser.add_argument('--config', type=str, default='configs/vanilla_gan.yaml',
                        help='Path to the config file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index, -1 for CPU')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    config["experiment"]["device"] = str(device)
    
    # Set random seed for reproducibility
    seed = config["experiment"].get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create dataloader
    dataloader = create_dataloader(config)
    
    # Create model
    model = VanillaGAN(config)
    
    # Create trainer and train
    trainer = VanillaGANTrainer(model, config, dataloader)
    trainer.train()


if __name__ == "__main__":
    main()
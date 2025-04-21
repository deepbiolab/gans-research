"""
Training script for DCGAN on MNIST dataset.
"""

import argparse
import yaml

from src.models import DCGAN
from src.losses import VanillaGANLoss
from src.training import GANTrainer
from src.data.dataloader import create_dataloader
from src.utils.set_experiment import configure_experiment


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
    dataloaders = create_dataloader(config)
    train_dataloader, valid_dataloader = dataloaders

    # Create model
    model = DCGAN(config)

    # Create loss function
    loss_fn = VanillaGANLoss()

    # Create trainer and train
    trainer = GANTrainer(
        model, config, train_dataloader, valid_dataloader, loss_fn=loss_fn
    )
    trainer.train()


if __name__ == "__main__":
    main()

"""
Utility functions for experiment environment configuration, including device and random seed setup.
"""

import os
import logging
from typing import Union, Tuple, Optional, List, Any
import wandb
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .visualization import make_grid


def setup_device(gpu_id: int = -1) -> Tuple[torch.device, str]:
    """
    Set up and return the appropriate device for computation.

    Args:
        gpu_id: GPU index to use. If -1 or if CUDA is not available, CPU will be used.

    Returns:
        tuple: (device, device_name)
            - device: torch.device object
            - device_name: string description of the device
    """
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def set_random_seed(seed: int, use_cuda: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Integer seed for random number generators
        use_cuda: Whether to also set CUDA seeds and settings
    """
    # Set Python's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed
    torch.manual_seed(seed)

    if use_cuda and torch.cuda.is_available():
        # Set CUDA random seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_experiment(
    config: dict, gpu_id: int = -1, seed: Union[int, None] = None
) -> Tuple[torch.device, str]:
    """
    Configure experiment environment including device and random seed.

    Args:
        config: Configuration dictionary
        gpu_id: GPU index to use. If -1 or if CUDA is not available, CPU will be used
        seed: Random seed. If None, will use seed from config

    Returns:
        tuple: (device, device_name)
            - device: torch.device object
            - device_name: string description of the device
    """
    # Set up device
    device = setup_device(gpu_id)
    config["experiment"]["device"] = str(device)

    # Set random seed
    if seed is None:
        seed = config["experiment"].get("seed", 42)  # Default to 42 if not specified
    set_random_seed(seed)

    return config


def setup_logger(
    output_dir: str,
    log_file: str = "training.log",
    level: int = logging.INFO,
    format_str: str = "%(asctime)s [%(levelname)s] %(message)s",
    handlers: Optional[List] = None,
) -> logging.Logger:
    """
    Setup logging configuration for training.

    Args:
        output_dir: Directory to save log file
        log_file: Name of the log file
        level: Logging level
        format_str: Format string for log messages
        handlers: Optional list of handlers to use instead of default ones

    Returns:
        Logger instance
    """

    if handlers is None:
        handlers = [
            logging.FileHandler(os.path.join(output_dir, log_file)),
            logging.StreamHandler(),
        ]

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
    )

    logger = logging.getLogger(__name__)
    return logger


class WandbWriter:
    """Create a wrapper class with TensorBoard-like interface for wandb"""

    def add_scalar(self, name, value, step):
        """Add a scalar value to the writer."""
        wandb.log({name: value}, step=step)

    def add_images(self, name, img_tensors, step, dataformats="NCHW", nrow=8):
        """
        Add multiple images to the writer.

        Args:
            name: Name of the image
            img_tensors: List or tensor of image tensors
            step: Global step value to record
            dataformats: Data format of the image tensor, this will used in tensorboard `add_images`
            nrow: Number of images displayed in each row of the grid
        """
        # Create a grid of images to control layout
        if img_tensors.dim() == 4:  # (batch, channels, height, width)
            # Create a grid with specified number of images per row
            grid = make_grid(img_tensors, nrow=nrow, normalize=True, padding=2)
            # Log as a single image to maintain the grid layout
            wandb.log({name: wandb.Image(grid)}, step=step)
        else:
            # Fallback for non-standard tensor shapes
            images = [wandb.Image(img) for img in img_tensors]
            wandb.log({name: images}, step=step)

    def add_artifact(self, artifact_name, artifact_type, file_path, aliases=None):
        """Add an artifact to the writer."""
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(file_path)
        wandb.log_artifact(artifact, aliases=aliases or [])

    def use_artifact(self, artifact_name, artifact_type=None, alias="latest"):
        """
        Download and use an artifact from wandb.

        Args:
            artifact_name (str): The name of the artifact, e.g., "project/model"
            type (str, optional): The type of the artifact (e.g., "model", "dataset"). Default is None.
            alias (str, optional): The alias or version to use (e.g., "latest", "v1"). Default is "latest".

        Returns:
            artifact_dir (str): Local directory where the artifact is downloaded.
            artifact (wandb.Artifact): The artifact object itself.
        """
        # Compose artifact reference string
        artifact_ref = f"{artifact_name}:{alias}" if alias else artifact_name
        artifact = wandb.use_artifact(artifact_ref, type=artifact_type)
        artifact_dir = artifact.download()
        return artifact_dir, artifact

    def close(self):
        """Close the writer"""
        wandb.finish()


def setup_summary(config: dict, output_dir: str) -> Any:
    """
    Setup experiment tracking with either TensorBoard or Weights & Biases (wandb).

    Args:
        config: Configuration dictionary containing logging settings
        output_dir: Directory to save logs

    Returns:
        A writer object for logging metrics, or None if logging is disabled
    """
    wandb_experiment = config.get("experiment", {}).get("name", None)
    logging_config = config.get("logging", {})

    # Setup wandb if specified
    if logging_config.get("use_wandb", False):
        # Get wandb configuration
        wandb_project = logging_config.get("wandb_project", "gan-research")
        wandb_task = logging_config.get("wandb_task", "train")
        wandb_config = logging_config.get("wandb_config", config)

        # Initialize wandb
        wandb.init(
            project=wandb_project,
            group=wandb_experiment,
            job_type=wandb_task,
            config=wandb_config,
            dir=output_dir,
        )

        return WandbWriter()

    # Setup TensorBoard if specified or as fallback
    if logging_config.get("use_tensorboard", True):
        return SummaryWriter(os.path.join(output_dir, "logs"))

    # No logging enabled
    return None

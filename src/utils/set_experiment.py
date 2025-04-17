"""
Utility functions for experiment environment configuration, including device and random seed setup.
"""

from typing import Union, Tuple
import numpy as np
import torch


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

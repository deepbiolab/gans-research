"""
Data augmentation and utility functions for GAN training.

This module provides:
1. DiffAugment implementation with various augmentation policies
2. Label generation utilities for label smoothing and random flipping
3. Helper functions for manipulating tensors during training
"""

from typing import List, Dict, Callable
import torch
import torch.nn.functional as F


def augment(
    x: torch.Tensor, policy: str = "", channels_first: bool = True
) -> torch.Tensor:
    """
    Apply augmentation policies to a batch of images.

    The implementation follows the DiffAugment paper:
    "Differentiable Augmentation for Data-Efficient GAN Training"
    https://arxiv.org/abs/2006.10738

    Args:
        x: Input tensor of shape (B, C, H, W) if channels_first=True, or (B, H, W, C) otherwise
        policy: Comma-separated list of augmentation operations to perform
                Options are "color", "translation", "cutout"
        channels_first: Whether the input tensor has channels first (PyTorch format)
                       or channels last (TensorFlow format)

    Returns:
        Augmented tensor with the same shape as input

    Example:
        >>> images = torch.randn(8, 3, 64, 64)
        >>> augmented = augment(images, policy="color,translation")
    """
    if policy:
        # Convert to channels-first if needed
        if not channels_first:
            x = x.permute(0, 3, 1, 2)

        # Apply each augmentation policy
        for p in policy.split(","):
            # Skip if the policy is not defined
            if p not in AUGMENT_FNS:
                continue

            # Apply all functions for this policy
            for f in AUGMENT_FNS[p]:
                x = f(x)

        # Convert back to original format if needed
        if not channels_first:
            x = x.permute(0, 2, 3, 1)

        # Ensure contiguous memory for efficiency
        x = x.contiguous()
    return x


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly adjust brightness of images.

    Adds a random value in range [-0.5, 0.5] to all pixels.

    Args:
        x: Input tensor of shape (B, C, H, W)

    Returns:
        Brightness-adjusted tensor of the same shape
    """
    # Generate random brightness adjustment factor for each image in batch
    brightness_factor = (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    )
    return x + brightness_factor


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly adjust saturation of images.

    Interpolates between grayscale and original image with a random factor.

    Args:
        x: Input tensor of shape (B, C, H, W)

    Returns:
        Saturation-adjusted tensor of the same shape
    """
    # Calculate mean across channels (grayscale equivalent)
    x_mean = x.mean(dim=1, keepdim=True)

    # Generate random saturation scaling factor (0-2) for each image
    sat_factor = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2

    # Adjust saturation: x_result = (x - gray) * sat_factor + gray
    return (x - x_mean) * sat_factor + x_mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly adjust contrast of images.

    Scales the difference from mean by a random factor in range [0.5, 1.5].

    Args:
        x: Input tensor of shape (B, C, H, W)

    Returns:
        Contrast-adjusted tensor of the same shape
    """
    # Calculate mean across all dimensions except batch
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)

    # Generate random contrast scaling factor (0.5-1.5) for each image
    contrast_factor = (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    )

    # Adjust contrast: x_result = (x - mean) * contrast_factor + mean
    return (x - x_mean) * contrast_factor + x_mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """
    Randomly translate images.

    Shifts the image by a random amount within the specified ratio of image dimensions.

    Args:
        x: Input tensor of shape (B, C, H, W)
        ratio: Maximum translation as a fraction of image size

    Returns:
        Translated tensor of the same shape
    """
    # Calculate maximum shift in pixels based on ratio
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

    # Generate random translation amounts for each image in batch
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    )
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    )

    # Create grid indices for each pixel in each image
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )

    # Shift grid by translation amounts and clamp to valid range
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)

    # Pad input tensor to handle boundary translations
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])

    # Remap pixels using the shifted grid
    x = (
        x_pad.permute(0, 2, 3, 1)  # BCHW -> BHWC for indexing
        .contiguous()[grid_batch, grid_x, grid_y]  # Perform remapping
        .permute(0, 3, 1, 2)  # BHWC -> BCHW
        .contiguous()
    )
    return x


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """
    Randomly cut out rectangles from images.

    Creates a random rectangular mask for each image and sets those regions to zero.

    Args:
        x: Input tensor of shape (B, C, H, W)
        ratio: Size of cutout as a fraction of image dimensions

    Returns:
        Tensor with random regions cut out
    """
    # Calculate cutout size in pixels based on ratio
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

    # Generate random offset for the cutout in each image
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )

    # Create grid indices for the cutout region
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )

    # Adjust grid to be centered at the random offset
    grid_x = torch.clamp(
        grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
    )

    # Create mask (1s everywhere)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)

    # Set cutout regions to 0 in the mask
    mask[grid_batch, grid_x, grid_y] = 0

    # Apply mask to input
    x = x * mask.unsqueeze(1)
    return x


# Dictionary mapping augmentation policy names to their functions
AUGMENT_FNS: Dict[str, List[Callable[[torch.Tensor], torch.Tensor]]] = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def get_positive_labels(
    size: int, device: torch.device, smoothing: bool = True, random_flip: float = 0.05
) -> torch.Tensor:
    """
    Generate labels for real data with optional smoothing and random flipping.

    Args:
        size: Number of labels to generate
        device: Device to create tensor on
        smoothing: Whether to apply label smoothing (values between 0.8-1.2 instead of 1.0)
        random_flip: Probability of flipping a positive label to 0

    Returns:
        Tensor of positive labels with smoothing and/or flipping applied

    Example:
        >>> real_labels = get_positive_labels(64, device, smoothing=True, random_flip=0.1)
    """
    if smoothing:
        # Random positive numbers between 0.8 and 1.2 (label smoothing)
        labels = 0.8 + 0.4 * torch.rand(size, device=device)
    else:
        labels = torch.full((size,), 1.0, device=device)

    if random_flip > 0:
        # Let's flip some of the labels to make it slightly harder for the discriminator
        num_to_flip = int(random_flip * labels.size(0))

        # Get random indices and set the first "num_to_flip" of them to 0
        indices = torch.randperm(labels.size(0))[:num_to_flip]
        labels[indices] = 0

    return labels


def get_negative_labels(size: int, device: torch.device) -> torch.Tensor:
    """
    Generate labels for fake data (all zeros).

    Args:
        size: Number of labels to generate
        device: Device to create tensor on

    Returns:
        Tensor of zeros of specified size

    Example:
        >>> fake_labels = get_negative_labels(64, device)
    """
    return torch.full((size,), 0.0, device=device)

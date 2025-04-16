"""
Visualization utilities for qualitative assessment of models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from src.models.base.base_generator import BaseGenerator


def make_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: tuple = (-1, 1),
) -> torch.Tensor:
    """
    Make a grid of images.
    Args:
        images: Tensor of images, (B, C, H, W)
        nrow: Number of images displayed in each row of the grid
        padding: Amount of padding
        normalize: If True, shift the image to the range (0, 1)
        value_range: Range of input images, needed for normalization
    Returns:
        grid: Grid of images, (C, H * nrow, W * ncol)
    """
    grid = torchvision.utils.make_grid(
        images,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )
    return grid


def save_image_grid(grid: torch.Tensor, filepath: str) -> None:
    """Save a grid of images."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torchvision.utils.save_image(grid, filepath)
    return


def display_image_grid(
    grid: torch.Tensor, title: str = None, figsize: tuple = (10, 10)
):
    """Display a grid of images."""

    # Convert tensor to numpy array
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().detach().numpy()

    # Convert from CHW to HWC format
    grid = np.transpose(grid, (1, 2, 0))

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(grid)

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return grid


def visualize_training_progress(
    generator: BaseGenerator,
    device: torch.device,
    fixed_noise: torch.Tensor,
    output_dir: str,
    iteration: int,
    nrow: int = 8,
    image_name: str = "progress",
) -> torch.Tensor:
    """
    Generate and save images using fixed noise vectors to visualize training progress.

    Args:
        generator: Generator model
        device: Device to run generation on
        fixed_noise: Fixed latent vectors for consistent visualization
        output_dir: Directory to save images
        iteration: Current iteration (used for filename)
        nrow: Number of images per row
        image_name: Base name for the saved image
    """
    generator.eval()
    with torch.no_grad():
        # Move fixed_noise to device
        fake_images = generator(fixed_noise.to(device))
    generator.train()

    # Create output directory if it doesn't exist
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Save grid of generated images
    filepath = os.path.join(samples_dir, f"{image_name}_{iteration:06d}.png")

    grid = make_grid(fake_images, nrow=nrow)
    save_image_grid(grid, filepath)

    return fake_images


def save_interpolation_grid(
    generator: BaseGenerator,
    device: torch.device,
    start_vector: torch.Tensor,
    end_vector: torch.Tensor,
    steps: int = 10,
    output_path: str = None,
    nrow: int = None,
):
    """
    Generate and save an interpolation between two latent vectors.

    Args:
        generator: Generator model
        device: Device to run generation on
        start_vector: Starting latent vector
        end_vector: Ending latent vector
        steps: Number of interpolation steps
        output_path: Path to save the interpolation grid
        nrow: Number of images per row (default: all in one row)
    """
    generator = generator.to(device)
    generator.eval()

    # Ensure vectors are on the correct device
    start_vector = start_vector.to(device)
    end_vector = end_vector.to(device)

    # Generate intermediate vectors
    alpha_values = np.linspace(0, 1, steps)
    vectors = []

    for alpha in alpha_values:
        interpolated = start_vector * (1 - alpha) + end_vector * alpha
        vectors.append(interpolated)

    # Stack all vectors (assume vectors are [1, latent_dim])
    all_vectors = torch.cat(vectors, dim=0)

    # Generate images
    with torch.no_grad():
        images = generator(all_vectors)

    # Set default number of rows if not provided
    if nrow is None:
        nrow = steps

    grid = make_grid(images, nrow=nrow)

    # Save or return the grid
    if output_path is not None:
        save_image_grid(grid, output_path)
        return

    grid = display_image_grid(grid, title="Latent Space Interpolation")

    return grid, images

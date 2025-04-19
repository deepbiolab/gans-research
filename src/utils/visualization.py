"""
Visualization utilities for qualitative assessment of models.
"""

import os
import glob
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision


def make_grid(
    samples: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: tuple = (-1, 1),
) -> torch.Tensor:
    """
    Make a grid of images.
    Args:
        samples: Tensor of samples, (B, C, H, W)
        nrow: Number of images displayed in each row of the grid
        padding: Amount of padding
        normalize: If True, shift the image to the range (0, 1)
        value_range: Range of input images, needed for normalization
    Returns:
        grid: Grid of images, (C, H * nrow, W * ncol)
    """
    grid = torchvision.utils.make_grid(
        samples,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )
    return grid


def save_grid(grid: torch.Tensor, filepath: str) -> None:
    """Save a grid of images."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert tensor to numpy array
    grid = grid.cpu().detach().numpy()
    
    # Convert from CHW to HWC format
    grid = np.transpose(grid, (1, 2, 0))

    # Convert to PIL image for resizing
    if grid.shape[2] == 1:  # Grayscale
        grid = Image.fromarray((grid[:, :, 0] * 255).astype(np.uint8), mode="L")
    else:  # RGB
        grid = Image.fromarray((grid * 255).astype(np.uint8))

    # Resize grid
    width, height = grid.size
    new_width = int(width * 2.0)
    new_height = int(height * 2.0)
    grid_resized = grid.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized grid
    grid_resized.save(filepath, dpi=(300, 300), optimize=True)


def display_grid(
    grid: torch.Tensor, title: str = None, figsize: tuple = (10, 10)
):
    """Display a grid of images."""

    # Convert tensor to numpy array
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


def extract_number(filename):
    """Extract the iteration number from a filename."""
    match = re.search(r"(\d+)\.png$", filename)
    if match:
        return int(match.group(1))
    return 0


def create_animation(
    experiment_dir: str,
    samples_subdir: str = "samples",
    pattern: str = "progress_*.png",
    duration: int = 200,
    include_iteration_text: bool = True,
) -> str:
    """
    Create a animation from training progress images.

    Args:
        experiment_dir: Root directory of the experiment
        samples_subdir: Subdirectory containing sample images
        pattern: Glob pattern to match image files
        duration: Duration of each frame in milliseconds (for GIF)
        include_iteration_text: Whether to add iteration number as text overlay

    Returns:
        Path to the created animation file
    """

    # Construct paths
    samples_dir = os.path.join(experiment_dir, samples_subdir)
    output_path = os.path.join(experiment_dir, "training_animation.gif")

    # Find all matching image files
    image_files = glob.glob(os.path.join(samples_dir, pattern))

    if not image_files:
        raise ValueError(
            f"No images found matching pattern '{pattern}' in {samples_dir}"
        )

    # Sort the image files by iteration number
    image_files.sort(key=extract_number)

    # Load images
    images = []
    for file in image_files:
        img = Image.open(file)

        # Add iteration text if requested
        if include_iteration_text:
            iteration = extract_number(file)
            draw = ImageDraw.Draw(img)

            # Try to get a font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial", 20)
            except IOError:
                font = ImageFont.load_default()

            # Add text at the bottom of the image
            text = f"Iteration: {iteration}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            position = ((img.width - text_width) // 2, img.height - text_height - 10)

            # Draw text with black outline for better visibility
            for offset in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                draw.text(
                    (position[0] + offset[0], position[1] + offset[1]),
                    text,
                    fill="black",
                    font=font,
                )
            draw.text(position, text, fill="white", font=font)

        images.append(img)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
        quality=90,
    )

    return output_path
